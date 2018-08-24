#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue July 3

@author: pyzhang
"""
# July 27
# In Nahmias review paper, he mentioned that numerical experiments show that
# a two-dimensional decision space is good: c and kl
# where c is the critical number for order-up-to
# k is the critical shelf life threshold, 
# so we only sum up inventory with >=k life left
# This illuminates to a new direction: accounting for the *near* downstream,
# nodes only, and only the fresh inventories.
#
# Therefore, this intuition about k should also carry through the linear policy
# which is parameterized by beta.
# Let me try to come up with a base_vis policy that looks like that
#
# Result: tested variations of base policy, verified Nahmias' point.
#
# Next step:
# We actually have another lever! We are now using a certain type of threshold
# issuing policy, which considers the downstream travel time.
# This alone buys us a lot of benefit
# That is probably the key benefit of visibility: 

import numpy as np
import logging
from scipy import stats
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import datetime

seed = 11
np.random.seed(seed)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

logger.handlers = []

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('log.txt', mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

#logger.debug('This is a test debug message.')
#logger.info('This is a test info message.')
#logger.warning('This is a warning message.')
#logger.error('This is an error message.')
#logger.critical('This is a critical message.')

###############################################################################
############################ Function Definitions #############################
###############################################################################

# Update system state.
#
#
# Events order: 
# Initial code (July 8)
#   1. closest pipeline inventory arriving
#   2. pipeline inventory aging and moving closer
#   3. inflow from upstream to pipeline inventory
#   4. production arrival
#   5. outflow: demand fulfillment
#   6. outflow: shipment to downstream
#   7. disposal
#   8. on-hand inventory aging
# Updated sequence (July 11)
#   Following the sequence of events in Chen, Pang, and Pan (2014)
#   a. Pipeline inventory updating, expiring, arriving on hand, 
#   b. Order placement (up to b), propagation from downstream to upstream
#      this initiates flows from a supply node to the pipeline of downstream
#   d. Demand realization and fulfillment (by source nodes), production arrival
#   e. Oldest inventory expire, and on-hand inventory aging
#       
# Assumptions: 
#   Source nodes receives production right away. Therefore, source nodes don't
#       have pipeline inventory
#   inv and pipe are not negative after update -- this is guaranteed by input
#       this function is just for
#       updating, not for deciding the issuing policy etc.
#   length of 3rd dimension of inv = m, max life time
#   lead time on arc (i,j) is fixed
#   Waste does not include pipeline waste -- it's parent's responsibility
# Input:
#   n: total number of nodes in the network
#   m: max shelf life of product
#   A[i,j]: 1 if i can feed j, 0 otherwise
#   lead[i,j]: integer, lead time between i and j
#   inv[i,k]: amount of life-k inv for node i (on-hand)
#   pipe[l,i,k]: in-transit inventory, arriving in l periods
#   prod[i,k]: production of life-k inv at source node i 
#   flow[i,j,k]: flow quantity of life-k inv from i to j 
#   fill[i,k]: amount of life-k inv at node i that is shipped out
#   disp[i,k]: amount of life-k inv to dipose at node i
# Return:   
#           on-hand inventory, 
#           pipeline inventory
#           total waste (expiration+disposal)
#
def update(n, m, A, lead, inv, pipe, prod, flow, fill, disp, source, sink):
    waste = 0
    
    negTolerance = -0.001

    logging.debug("Inventory on hand:\n %s", str(inv))
    logging.debug("Pipeline inventory pipe[l,i,k]:\n %s \n", str(pipe))
    
    # 1. closest pipeline inventory arriving
    inv += pipe[0]
    logging.debug("Inventory update: received from pipeline\n %s \n", str(inv))
    if np.min(inv[1:,:]) < negTolerance:
        logging.warning("Inv below 0 after receiving pipe: \n %s", str(inv))
    
    # 2. pipeline inventory aging and moving closer
    pipe = np.roll(pipe, -1, axis=0) # aging
    pipe[-1:,:,:] = 0
    pipe = np.roll(pipe, -1, axis=2) # expiring
    # waste += sum3(pipe[:,:,-1:])
    pipe[:,:,-1:] = 0
    logging.debug("Pipeline update: aging\n %s \n", str(pipe))
    if np.min(pipe[:,1:,:]) < negTolerance:
        logging.warning("Pipe below 0 after pipe aging: \n %s", str(pipe))
    
    # Orders are transformed into flow vectors
    # 3a. inflow from upstream to pipeline inventory
    rows, cols = np.nonzero(A)
    for i in range(len(rows)):
        l = lead[rows[i],cols[i]]
        pipe[l-1][cols[i]] += np.sum(flow[:,cols[i],:],axis=0)
        #pipe[l-1] += np.sum(flow,axis=0)
    logging.debug("Pipeline update: receiving from upstream\n %s \n",str(pipe))
    if np.min(pipe[:,1:,:]) < negTolerance:
        logging.warning("Pipe below 0 after receiving flow: \n %s", str(pipe))
    
    # 3b. outflow: shipment to downstream
    inv -= np.sum(flow,axis=1)
    logging.debug("Inventory update: sending to downstream\n %s \n", str(inv))
    if np.min(inv[1:,:]) < negTolerance and np.argmin(inv) >= m:
        logging.warning("Inv below 0 after outflow: \n %s", str(inv))
    
    # 4. production arrival 
    inv += prod
    logging.debug("Productions:\n %s \n", str(prod))
    logging.debug("Inventory udpate: production arrival \n %s \n", str(inv))
    
    # 5. outflow: demand fulfillment (at sink nodes)
    inv -= fill
    logging.debug("Demand quantity to serve:\n %s \n", str(fill))
    logging.debug("Inventory update: demand fulfillment\n %s \n", str(inv))
    if np.min(inv[1:,:]) < negTolerance:
        logging.warning("Inv below 0 after demand fulfill: \n %s", str(inv))
    
    # 6. disposal
    inv -= disp
    waste += sum2(disp)
    logging.debug("Disposal:\n %s \n", str(disp))
    logging.debug("Inventory update: disposal\n %s \n", str(inv))
    if np.min(inv[1:,:]) < negTolerance:
        logging.warning("Inv below 0 after disposal: \n %s", str(inv))
    
    # 7. on-hand inventory aging
    
    # Assumptions: the source nodes are just holders for inv, wastes there 
    # are not included. We only include wastes on things that non-source nodes
    # ordered.
    #waste += sum2(inv[:,-1:])
    for i in range(n):
        if source[i] == 0:
            waste += inv[i,0]
    inv = np.roll(inv, -1, axis=1)
    inv[:,-1:] = 0
    logging.debug("Inventory update: aging\n %s \n", str(inv))

    return inv, pipe, waste


# Basic order-up-to for each node. Order up to b from closest supplier.
#
# Assumptions:
#   We include all inventories, including those might expire in transit 
#   
# Input:
#   n: number of nodes
#   A[i,j]: 1 if i can feed j, 0 otherwise
#   j: current node index
#   lead[i,j]: integer, lead time between i and j
#   inv[k] inventory of life-k
#   pipe[l,k] inventory of life-k that to arrive in l periods
#   d: demand current period
#   b: base level to hold
#
# Return:
#   orders: orders[i] quantity of inv to be ordered from i  
#     
def baseOrder(n, m, j, A, lead, inv, pipe, onOrder, b, beta, cl):
    usefulPipe = sum2(pipe)
    for l in range(np.minimum(m-1,np.shape(pipe)[0])):
        usefulPipe -= sum(pipe[l,:np.minimum(l+1+cl,np.shape(pipe)[1])])
    logging.debug("Total on-hand inv: %d",sum(inv))
    logging.debug("Total useful pipeline inventory: %d", usefulPipe)
    
    orders = np.zeros(n)
    toOrder = b - sum(inv[cl:]) - usefulPipe - sum(onOrder)
    if toOrder > 0:       
        i = np.argmin(lead[:,j])
        orders[i] = toOrder * beta

    return orders


# K-step visible order-up-to for each (non-sink) node. 
# Order up to b from closest supplier.
#
# Assumptions:
#   All sink nodes face i.i.d. demand random variable (how b is calculated
#
# Input:
#   n: number of nodes
#   A[i,j]: 1 if i can feed j, 0 otherwise
#   j: current node index
#   lead[i,j]: integer, lead time between i and j
#   inv[i,k] inventory of life-k at node i
#   pipe[l,i,k] inventory of life-k that to arrive in l periods
#   d: demand current period
#   b[i]: base level to hold for each node (in myopic case). Maybe not used.
#   K: K-step visible. 1 step means the node cares about 1 layer down
#
# Return:
#   orders: orders[i] quantity of inv to be ordered from i  
#     
def baseOrderVis(n, m, j, A, lead, inv, pipe, dparams, b, beta, cl, sl, K):
    invTarget = 0.0 # base level for K-steps
    invPosition = 0.0 # total useful inventory in the path

    invTarget, invPosition = baseDownstream(n,m,j,A,lead,dparams,inv,pipe,sl,K)
    toOrder = invTarget - invPosition

    orders = np.zeros(n)    
    if toOrder > 0:       
        i = np.argmin(lead[:,j])
        orders[i] = toOrder * beta

    return orders

def baseDownstream(n, m, j, A, lead, dparams, inv, pipe, sl, K):
    invTarget = 0.0
    invPosition = 0.0
    l = np.amin(lead,0) # min lead time for each node
    mean = dparams[0]
    std = dparams[1]
    
    if K < 0:
        return invTarget, invPosition
    else:
        usefulPipe = sum2(pipe[:,j,:])
        for ld in range(np.minimum(m-1,np.shape(pipe)[0])):
            usefulPipe -= sum(pipe[ld,j,:ld+1])
        onHand = sum(inv[j])
        invPosition = usefulPipe + onHand
        invTarget = np.maximum(0,(l[j]+1) * mean + \
                               np.sqrt(l[j]+1) * stats.norm.ppf(sl) * std)
        
        if K > 0:
            p = 0.0
            t = 0.0
            for downIdx in range(n):
                if A[j,downIdx] == 1:
                    t, p = baseDownstream(n,m,j,A,lead,dparams,inv,pipe,sl,K-1)
            invPosition += p
            invTarget += t
            
    return invTarget, invPosition
        
# 1. Generate (random) demands and yields
# 2. Decide nominal orders of every node, split orders to upstream, 
#      Compare these nominal orders and upstream capacities and inv limit
#      Decide the actual realized orders   
#
# Assumptions:
#   Nodes are strictly divided into levels. This simplifies the propagation 
#   of demand.
#   FIFO issuing except for source. Source sends freshest.
#   Production arrives at source nodes (push), other flows are pulled by orders
#   Each sink node faces the same demand (matters when setting base level 
#   for the non-sink and non-source nodes)
#    
# Input: n, A, 
#   lead[i,j]: lead time between i and j. Use 1000 to denote absence of link
#   inv[i,k]: amount of life-k inv for node i (on-hand)
#   pipe[l,i,k]: in-transit inventory, arriving in l periods
#   dparams: parameters to generate random demands [0] mean, [1] standard dev
#   yparams: parameters to generate random yields/capacities, [0] mean, [1] sd
#   sl: service level requirement
#   source: vector of 0 and 1 indicating whether a node is a farm (source)
#   sink: vector of 0 and 1 indicating whether a node is a retailer (sink)
#   sl: service level
#   orderHistory[i,j,t]: the order placed by j to i at time period t.
#                       used to calculate empirical demand at upstream    
#   leadAware: if True, issuing inventories that can only be fresh when arrive
#   at downstream. If False, then FIFO from oldest inventory.
#   fulfilled[i] the amount of order coming from i that is fulfilled
#   unfulfilled[i] the amount of order coming from i that is not fulfilled
#    
# Return: 
#   Total waste in the network
#   Demand fulfillment vector in the network
#   Demand loss
#    
def pull(policy, beta, cl, t, n, m, A, lead, inv, pipe, dparams, orderHistory, \
         yparams, source, sink, sl, K, leadAware, base, slvec, flowHistory, \
         onOrder, d):  
    waste = 0
    loss = 0
    #d = np.zeros(n)
    fulfilled = np.zeros(n)
    unfulfilled = d
    
    # Initialize a random d
    #d = np.maximum(0, np.random.normal(dparams[0], dparams[1], n))
    #d = np.minimum(d, sink * (dparams[0]+dparams[1]*2))
    
    logging.debug("Demand vector:\n %s \n", str(d))
    
    # Initialize yield vector
    prod = np.zeros((n,m))
    prod[:,m-1] = np.random.normal(yparams[0], yparams[1], n)
    prod[:,m-1] = np.minimum(prod[:,m-1], source * (yparams[0]+yparams[1]*10))
    
    logging.debug("Production vector:\n %s \n", str(prod))
    
    # Set base stock level based on service level, lead time, every node
    logging.debug("Lead time matrix:\n %s \n", str(lead))
    l = np.amin(lead,0) # min lead time for each node
    
    ## Calculate the minimum freshness a node i needs to send to downstream
    # Roughly speaking, we use the notion of shortest path to sink nodes
    toSink = np.ones(n,dtype=np.int16) * 1000
    front = np.copy(sink) 
    for i in range(n):
        if sink[i] == 1:
            toSink[i] = 0
    while front.any() > 0:
        copyFront = np.copy(front)
        for j in np.nonzero(copyFront)[0]:
            front[j] = 0
            for i in np.nonzero(A[:,j])[0]:
                front[i] = 1
                toSink[i] = min(toSink[i], toSink[j] + lead[i,j])
                
    # Directly take base stock policy
#    b = base
    b = np.zeros(n)
    for i in range(n):
        if sink[i] == 1:
            mean = dparams[0]
            std = dparams[1]
            b[i] = np.maximum(0,(l[i]+1) * mean + \
                           np.sqrt(l[i]+1) * stats.norm.ppf(slvec[i]) * std)
            #print "Retailer base level: ", b[i]
            #print "Retailer ordering: ", orderHistory[i-1,:,:t].sum(axis=1)
        # for source nodes (Farm), they just produce
        # therefore base stock for them is meaningless, and I set it to 0
        elif source[i] == 1:
            b[i] = 0
        else:
            if policy == "base":
                #mean = np.mean(flowHistory[i,:,:t].sum(axis=0))
                #std = np.std(flowHistory[i,:,:t].sum(axis=0))
                mean = np.mean(orderHistory[i,:,:t].sum(axis=0))
                std = np.std(orderHistory[i,:,:t].sum(axis=0))
                #print "Supplier sees order: ", mean, ", ", std
                #mean = dparams[0]
                #std = dparams[1]
                #print "Supplier received orders, mean: ", mean, ", stdev: ", std
            elif policy == "base_vis":
                mean = dparams[0]
                std = dparams[1]
            #mean = dparams[0]
            #std = dparams[1]
            b[i] = np.maximum(0,(l[i]+1) * mean + \
                          np.sqrt(l[i]+1) * stats.norm.ppf(slvec[i]) * std)
            #print "Supplier base level: ", b[i]
    
    logging.debug("Base level targets:\n %s \n", str(b))

    # Orders[i,j] is the total quantity ordered by j from i
    orders = np.zeros((n,n))
    # Figure out order quantity for the sink nodes
    front = np.copy(sink)
    for j in np.nonzero(sink)[0]:
        orders[:,j] = baseOrder(n,m,j,A,\
              lead,inv[j,:],pipe[:,j,:], onOrder[:,j],b[j], beta, cl).T        
    logging.debug("Orders: \n %s \n", str(orders))
    
    for j in np.nonzero(sink)[0]:
        front[j] = 0
        for i in np.nonzero(A[:,j])[0]:
            front[i] = 1    
    logging.debug("Orders initiated:\n %s \n", str(orders))
    
        
    # Aggregate raw orders to upstream as demand
    # propogate demand from sink to source
    while front.any() > 0:
        for j in np.nonzero(front)[0]:
            if source[j] == 0:
                if policy == "base":
                    orders[:,j] = baseOrder(n,m,j,A,
                                            lead,inv[j,:],pipe[:,j,:],onOrder[:,j],
                                            b[j], beta, cl)
                elif policy == "base_vis":
                    orders[:,j] = baseOrderVis(n,m,j,A,lead,inv,pipe, dparams,\
                          b, beta, cl, sl, K)
        copyFront = np.copy(front)
        for j in np.nonzero(copyFront)[0]:
            front[j] = 0
            for i in np.nonzero(A[:,j])[0]:
                front[i] = 1
    
    orderHistory[:,:,t] = orders
    
    # Converting orders[i,j] to flows[i,j,k]
    # Generate flow as much as possible, up to the upstream on-hand inv level
    # Assumptions: 
    #   FIFO but not counting old inv if aware of downstream leadtime
    #   Proportional fulfillment of orders
    #   Inv used only when there's enough life left to survive the lead time
    flow = np.zeros((n,n,m)) # flow[i,j,k]: amount of life-k flow from i to j
    for i in range(n):
        for j in range(n):
            if A[i,j] == 1:
                if source[i] == 1:
                    ## Farm always sends freshest
                    #flow[i,j,m-1] = orders[i,j] + onOrder[i,j]
                    ## Farm sends 3 buckets
                    r1 = np.random.rand() * 0.2 + 0.4
                    r2 = np.random.rand() * 0.1 + 0.2
                    r3 = 1 - r1 - r2
                    flow[i,j,m-1] = (orders[i,j] + onOrder[i,j]) * r1
                    flow[i,j,m-2] = (orders[i,j] + onOrder[i,j]) * r2
                    flow[i,j,m-3] = (orders[i,j] + onOrder[i,j]) * r3
                    ## Farm sends 2 buckets
                    #flow[i,j,m-1] = (orders[i,j] + onOrder[i,j]) / 5.0 * 3.0
                    #flow[i,j,m-2] = (orders[i,j] + onOrder[i,j]) / 5.0 * 2.0
                else:
                    if leadAware == False:
                        l = lead[i,j] + 1
                    else:
                        l = lead[i,j] + toSink[j] + 1
                    lastK = l
                    for k in range(m):
                        if k >= l:
                            if sum(inv[i,l:k+1])<=sum(orders[i,:]+onOrder[i,:]) \
                            and sum(orders[i,:]+onOrder[i,:]) > 0:
                                flow[i,j,l:k+1] = 1.0 * inv[i,l:k+1] * \
                                orders[i,j] / sum(orders[i,:]+onOrder[i,:])
                                lastK += 1
                    if lastK < m and lastK >= l and sum(orders[i,:]+onOrder[i,:]) > 0:
                        flow[i,j,lastK] = 1.0 * (sum(orders[i,:]+onOrder[i,:]) \
                        - sum(inv[i,l:lastK])) * (orders[i,j]+onOrder[i,j]) / sum(orders[i,:]+onOrder[i,:])
    
    #onOrder = onOrder + orders - flow.sum(axis=2)
    for i in range(n):
        for j in range(n):
            onOrder[i,j] = max(0, onOrder[i,j] - np.sum(flow[i,j,:]))
    
    flowHistory[:,:,t] = np.sum(flow, axis=2)
    
    logging.debug("Total orders for this period is %d", sum2(orders))
    logging.debug("Total flow for this period is %d", sum3(flow))
    
    # Transform d to fill array -- the actual demand fulfilled
    # fill[i,k] amount of life-k inventory being used by i to fulfill demand
    fill = np.zeros((n,m))
    for i in range(n):
        lastK = 0
        for k in range(m):
            if sum(inv[i,:k+1]) <= d[i]:
                fill[i,:k+1] = inv[i,:k+1]
                lastK += 1
        if lastK < m and d[i] > 0:
            fill[i,lastK] = d[i] - sum(inv[i,:lastK])
    logging.debug("Demand fulfilled:\n %s \n", str(fill))
    
    # Disposal decision
    # No intentionally disposal for this basic pull
    disp = np.zeros((n,m))
    
    # Call update function and accumulate waste, etc.
    logging.debug("Calling update function. \n")
    inv, pipe, newWaste = update(n, m, A, lead, inv, pipe, 
                                    prod, flow, fill, disp, source, sink)
    logging.debug("Finished update function. \n")
    
    waste += newWaste
    
    fulfilled = sum(fill.T).T
    unfulfilled = d - fulfilled
    loss = sum(unfulfilled)
    
    freshness = 0
    
    if sum(fulfilled) < 0.0001: # numerical hack for == 0
        freshness = 0
    else:
        freshness = np.dot(np.sum(fill,axis=0), np.arange(m)+1) / m / sum(fulfilled)
    
    logging.debug("Total demand filled: %d", sum2(fill))
    logging.debug("Total demand unfilled: %d", loss)
    
    logging.debug("Pipe \n %s ", str(pipe))
    
    return inv, pipe, waste, loss, freshness, fulfilled, unfulfilled, b, \
            orderHistory, d, flowHistory, onOrder

# Helper functions

def sum2(array):
    return sum(sum(array))

def sum3(array):
    return sum(sum(sum(array)))


###########################################################################
############################# Main Function ###############################
###########################################################################
    
T = 1000

dparams = np.array([10,7])
yparams = np.array([100,2])

dT = np.zeros(T)
for t in range(T):
    dT[t] = np.maximum(0, np.random.normal(dparams[0], dparams[1], 1))
    dT[t] = np.minimum(dT[t], (dparams[0]+dparams[1]*2))

numSL = 20
numSim = 5
numScenarios = 2
timePeriod = 1

saveFile = True
output = np.zeros((numSL,numSim,5,numScenarios))

for scenario in range(numScenarios):
    
    if scenario == 0:
        centralized = False
    elif scenario == 1:
        centralized = True
        
    longChain = False
    
    if longChain == False:
        if centralized == False:
            n = 3
            m = 10
            A = np.zeros((n,n))
            A[0,1] = 1
            A[1,2] = 1
            lead = np.zeros((n,n),dtype=np.int16) + 1000
            lead[0,1] = 3
            lead[1,2] = 3
            maxLead = 3
            totalLead = 6
            
            source = np.zeros(n)
            sink = np.zeros(n)
            source[0] = 1
            sink[2] = 1
            base = np.zeros(n)
            base[1] = 60
            base[2] = 60
            beta = 0.7
            cl = 0
            
        else:
            n = 2
            m = 10
            A = np.zeros((n,n))
            A[0,1] = 1
            lead = np.zeros((n,n),dtype=np.int16) + 1000
            lead[0,1] = 6
            maxLead = 6
            totalLead = 6
            
            source = np.zeros(n)
            sink = np.zeros(n)
            source[0] = 1
            base = np.zeros(n)
            sink[1] = 1
            base[1] = 110
            beta = 0.7
            cl = 0
    else:
        if centralized == False:
            n = 4
            m = 10
            A = np.zeros((n,n))
            A[0,1] = 1
            A[1,2] = 1
            A[2,3] = 1
            lead = np.zeros((n,n),dtype=np.int16) + 1000
            lead[0,1] = 2
            lead[1,2] = 2
            lead[2,3] = 2
            maxLead = 2
            totalLead = 6
            
            source = np.zeros(n)
            sink = np.zeros(n)
            source[0] = 1
            sink[3] = 1
            base = np.zeros(n)
            base[1] = 40
            base[2] = 40
            base[3] = 40
        else:
            n = 3
            m = 10
            A = np.zeros((n,n))
            A[0,1] = 1
            A[1,2] = 1
            lead = np.zeros((n,n),dtype=np.int16) + 1000
            lead[0,1] = 2
            lead[1,2] = 4
            maxLead = 4
            totalLead = 6
            
            source = np.zeros(n)
            sink = np.zeros(n)
            source[0] = 1
            sink[2] = 1
            base = np.zeros(n)
            base[1] = 60
            base[2] = 60
    
    

    
    print " ", 0, "% @ ", datetime.datetime.today()
    
    policy = "base"
    
    leadAware = True
    
    #policy = "base_vis" # obsolete for now. Visibility should go into the 
    K = 0 # K-step visibility into downstream inv and lead time information
    
    # For the linear policy in Nahmias: order (S - sum(inv))*\beta
    # where S is the non-perishable inventory order up to quantity
    #beta = 0.6 # choose 1 or 0.5
    #cl = 1 #  count all inv with shelflife >= cl. from 0 to m-1, in this case 0 or 1

    
    for it in range(numSL):
        for s in range(numSim):
            
            #sl = 1.0 * it / numSL 
            sl = 1.0 - np.exp(-1.0*it/3)
            
            # Temp: for demo
            if centralized == False:
                slvec = np.zeros(n)
                if longChain == False:
                    slvec[1] = sl  # longChain = false, dvar = 7
                    slvec[2] = sl # longChain = false, dvar = 7
                    
                    
                else:
                    slvec[1] = sl
                    slvec[2] = sl
                    slvec[3] = sl
            else:
                slvec = np.ones(n) * sl
                    
            inv = np.maximum(np.random.rand(n,m), 0)
            pipe = np.maximum(np.random.rand(maxLead, n, m), 0)
            onOrder = np.zeros((n,n)) # on-order but not into pipe yet.
                    
            orderHistory = np.zeros((n,n,T))
            # Now using flowHistory instead of orderHistory to construct 
            # demand at upstream, since in the lost demand model, supplier can 
            # witness repeated orders
            flowHistory = np.zeros((n,n,T))
            
            totalWaste = 0.0
            totalLoss = 0.0
            totalFreshness = 0.0
            baseStocks = np.zeros(n)
            
            logging.info("Initial inventory: \n %s \n", str(inv))
            logging.info("Initial pipeline: \n %s \n", str(pipe))
            
            # path at the demand node in the serial network
            invPath = np.zeros(T)
            invDistPath = np.zeros((m,T)) # At the retailer
            pipePath = np.zeros(T)
            demandPath = np.zeros(T)
            lossPath = np.zeros(T)
            wastePath = np.zeros(T)
            freshnessPath = np.zeros(T)
            
            # For August demo on the dynamics of system
            if centralized == True:
                c = "consensus"
            else:
                c = "independent"
                
            csvDemand = open("output/demand_"+str(n)+"_"+str(it)\
                             +"_"+str(c)\
                             +"_period_"+str(timePeriod)\
                             +".csv", "w") 
            csvOrder = open("output/order_"+str(n)+"_"+str(it)\
                            +"_"+str(c)\
                            +"_period_"+str(timePeriod)\
                            +".csv", "w")
            #csvFlow = open("output/flow_"+str(n)+"_"+str(it)\
            #               +"_centralized_"+str(centralized)\
            #               +"_period_"+str(timePeriod)\
            #               +".csv", "w")
            #csvLoss = open("output/loss_"+str(n)+"_"+str(it)\
            #               +"_centralized_"+str(centralized)\
            #               +"_period_"+str(timePeriod)\
            #               +".csv", "w")
            #csvPipe = open("output/pipe_"+str(n)+"_"+str(it)\
            #               +"_centralized_"+str(centralized)\
            #               +"_period_"+str(timePeriod)\
            #               +".csv", "w")
            csvInv = open("output/inv_"+str(n)+"_"+str(it)\
                          +"_"+str(c)\
                          +"_period_"+str(timePeriod)\
                          +".csv", "w")
            csvWaste = open("output/waste_"+str(n)+"_"+str(it)\
                            +"_"+str(c)\
                            +"_period_"+str(timePeriod)\
                            +".csv", "w")
            csvFresh = open("output/fresh_"+str(n)+"_"+str(it)\
                            +"_"+str(c)\
                            +"_period_"+str(timePeriod)\
                            +".csv", "w")
            
            currentDate = datetime.date.today()
            
            csvDemand.write("Date, Demand placed by, Demand \n")
            csvInv.write("Date, node, time since harvest, inventory quantity \n")
            csvWaste.write("Date, Waste \n")
            csvFresh.write("Date, Freshness \n")
            
            for t in range(T):
                d = np.zeros(n)
                d[n-1] = dT[t]
                inv, pipe, waste, loss, freshness, fulfilled, unfulfilled, \
                baseStocks, orderHistory, d, flowHistory, onOrder = \
                pull(policy, beta, cl, t, n, m, A, lead, inv, pipe, dparams, \
                     orderHistory, yparams, source, sink, sl, K, leadAware, \
                     base, slvec, flowHistory, onOrder, d)
                    
                invPath[t] = sum(inv[n-1,:])
                invDistPath[:,t] = inv[n-1,:]
                pipePath[t] = sum2(pipe[:,n-1,:])
                wastePath[t] = waste
                lossPath[t] = loss
                demandPath[t] = sum(fulfilled + unfulfilled)
                freshnessPath[t] = freshness * m/(m-totalLead)
                
                ignoreHead = 100
                if t > ignoreHead:
                    totalWaste += waste
                    totalLoss += loss
                    totalFreshness += freshness * m/(m-totalLead)
                          
                logging.debug("Service level, baseStock, total waste, total loss: \
                 %f, %f, %d, %d", sl, baseStocks[-1:], totalWaste, totalLoss)
                logging.debug("Inventory after pull %d: \n %s", t, str(inv))
                logging.debug("Pipeline after pull %d: \n %s", t, str(pipe))
                
                if t >= ignoreHead:
                    if (t-ignoreHead) % timePeriod == 0:
                        demandThisPeriod = d[n-1] 
                        orderThisPeriod = np.copy(orderHistory[:,:,t])
                        #flowThisPeriod = fulfilled
                        #lossThisPeriod = loss 
                        pipeThisPeriod = np.copy(pipe)
                        invThisPeriod = inv
                        wasteThisPeriod = waste
                        freshThisPeriod = freshness * m/(m-totalLead) / timePeriod
                    else:
                        demandThisPeriod += d[n-1]
                        orderThisPeriod += orderHistory[:,:,t]
                        #flowThisPeriod += fulfilled
                        #lossThisPeriod += loss
                        pipeThisPeriod += pipe
                        invThisPeriod += inv
                        wasteThisPeriod += waste
                        freshThisPeriod += freshness * m/(m-totalLead) / timePeriod                    
    
                    if (t-ignoreHead) % timePeriod == timePeriod - 1:
                        
                        #csvDemand.write(str(demandThisPeriod)+"\n")
                        csvDemand.write(str(currentDate)+",Demand by supplier,"+
                                        "{:.2f}".format(orderThisPeriod[0,1])+"\n")
                        csvDemand.write(str(currentDate)+",Demand by retailer,"+
                                        "{:.2f}".format(orderThisPeriod[n-2,n-1])+"\n")
                        csvDemand.write(str(currentDate)+",Demand by customer,"+
                                        "{:.2f}".format(demandThisPeriod)+"\n")
                        
                        csvOrder.write(str(orderThisPeriod)+"\n")
                        
                        #csvFlow.write(str(flowThisPeriod)+"\n")
                        #csvLoss.write(str(lossThisPeriod)+"\n")
                        #csvPipe.write(str(pipeThisPeriod)+"\n")
                        
                        ##csvInv.write(str(np.sum(invThisPeriod,axis=1))+"\n")
                        if centralized == False:
                            for i in range(1,n):                        
                                for k in range(m):
                                    if i == n-1:
                                        node = "retailer"
                                    else:
                                        node = "supplier"
                                    invHere = invThisPeriod[i,k]
                                    csvInv.write(str(currentDate)+","+
                                                 node+","+str(m-k-1)+","+
                                                 "{:.2f}".format(invHere)+"\n")
                        else:
                            # supplier. only one at the moment. hard coded
                            r = 1
                            sr_lead = 2
                            for k in range(m):
                                invHere = pipe[sr_lead,r,k]
                                csvInv.write(str(currentDate)+","+
                                             "supplier"+","+str(m-k-1)+","+
                                             "{:.2f}".format(invHere)+"\n")
                            # retailer inv is just from inv matrix
                            for k in range(m):
                                invHere = invThisPeriod[1,k]
                                csvInv.write(str(currentDate)+","+
                                             "retailer"+","+str(m-k-1)+","+
                                             "{:.2f}".format(invHere)+"\n")
                            
                        csvWaste.write(str(currentDate)+","+"{:.2f}".format(wasteThisPeriod)+"\n")
                        csvFresh.write(str(currentDate)+","+"{:.2f}".format(freshThisPeriod)+"\n")
                        currentDate += datetime.timedelta(days=1)
                    
            
            csvDemand.close()
            csvOrder.close()
            #csvFlow.close()
            #csvLoss.close()
            #csvPipe.close()
            csvInv.close()
            csvWaste.close()
            csvFresh.close()
            
            output[it, s, 0, scenario] = sl
            output[it, s, 1, scenario] = baseStocks[n-1]
            output[it, s, 2, scenario] = 1.0 * totalWaste / (T - ignoreHead)
            output[it, s, 3, scenario] = 1.0 * totalLoss / (T - ignoreHead)
            output[it, s, 4, scenario] = 1.0 * totalFreshness / (T - ignoreHead)
            
            print "Scenario ", scenario, ": ", 100.0 / numSL * (it + 1.0 * (s + 1) / numSim), \
                        "% @ ", datetime.datetime.today()
             
            
    # print trajectories
    ######################################################################
    ## Plot 2: demand, inventory, demand loss levels during a time window
    ######################################################################
    fig = plt.figure()
    plt.clf()
    ax1 = fig.add_subplot(111)
    window = np.minimum(1000, T-1)
    #ax1.scatter(range(T), demandPath, s=10, c='b', label='Demand')
    #ax1.scatter(range(T), lossPath, s=10, c='g', label='Loss')
    #ax1.scatter(range(T), invPath, s=10, c='k', label='Inv')
    #ax1.scatter(range(T), pipePath, s=10, c='r', label='Pipe')
    ax1.plot(range(min(T,window)), demandPath[-1*min(T,window):], c='royalblue', \
             label='Demand')
    ax1.plot(range(min(T,window)), lossPath[-1*min(T,window):], c='lightcoral', \
             label='Loss')
    ax1.plot(range(min(T,window)), invPath[-1*min(T,window)-1:-1], c='c', \
             label='Inv (R)')
    #ax1.plot(range(min(T,window)), invPath[-1*min(T,window)-1:-1]+\
    #lossPath[-1*min(T,window):], c='y', label='Inv+Loss')
    #ax1.plot(range(min(T,window)), pipePath[-1*min(T,window)-1:-1], c='y', \
    #         label='Pipe')
    plt.legend(loc='upper left');
    #plt.show()
    if saveFile:
        plt.savefig("output/trajectory_policy="+policy+"_beta=_"+str(beta)\
                    +"_cl="+str(cl)+"_n="+str(n)+"_life="+str(m)+"_maxLead="\
                    +str(maxLead)+"_leadAware="+str(leadAware)\
                    +"_seed="+str(seed)+"_horizon="\
                    +str(T)+"_numPol="+str(numSL)+"_numSim="+str(numSim)+"_"\
                    +str(datetime.datetime.today())+".eps", format='eps', dpi=1000)
        plt.show()
    else:
        plt.show()
        
    # Freshness plot
    fig = plt.figure()
    plt.clf()
    ax1 = fig.add_subplot(111)
    window = np.minimum(1000, T-1)
    ax1.plot(range(min(T,window)), freshnessPath[-1*min(T,window)-1:-1], c='b', \
             label='Freshness')
    plt.legend(loc='upper left');
    plt.show()
        
    # Waste plot
    fig = plt.figure()
    plt.clf()
    ax1 = fig.add_subplot(111)
    window = np.minimum(1000, T-1)
    ax1.plot(range(min(T,window)), wastePath[-1*min(T,window)-1:-1], c='g', \
             label='Waste')
    plt.legend(loc='upper left');
    plt.show()
    
    # Waste plot
    fig = plt.figure()
    plt.clf()
    ax1 = fig.add_subplot(111)
    window = np.minimum(1000, T-1)
    ax1.plot(range(min(T,window)), lossPath[-1*min(T,window)-1:-1], c='c', \
             label='Demand loss')
    plt.legend(loc='upper left');
    plt.show()
    
            #print "Life distribution"            
            #print np.sum(invDistPath,axis=1)/T
            
        #print 100.0 * (it+1) / numSL, "% @ ", datetime.datetime.today()
    
    print output[:,:,:,scenario]
    
    # Colors: https://i.stack.imgur.com/lFZum.png

if saveFile:
    outputFile = "output/output_policy="+policy+"_beta=_"+str(beta)\
    +"_cl="+str(cl)+"_n="+str(n)\
    +"_life="+str(m)\
    +"_maxLead="+str(maxLead)+"_leadAware="+str(leadAware)\
    +"_seed="+str(seed)\
    +"_horizon="+str(T)+"_numPol="+str(numSL)+"_numSim="+str(numSim) \
    +"_"+str(datetime.datetime.today())
    np.save(outputFile, output)
    
    
sColors = []
sColors.append('gray')
sColors.append('blue')

########################################################
## Plot 3: the Pareto front of demand loss vs freshness
########################################################
for sc in range(numScenarios):
    plt.scatter(output[:,:,4,sc].sum(axis=1)/numSim, \
                output[:,:,3,sc].sum(axis=1)/numSim, \
                c = sColors[sc], s = 10)
#plt.plot(output[:,2], output[:,3], '-')
#plt.xlim(0, 30)
#plt.ylim(0, 12)
#plt.title('TITLE', fontsize=20)
plt.xlabel('Average freshness', fontsize=15)
plt.ylabel('Average demand loss', fontsize=15)
plt.gray()
ax = plt.gca()
ax.set_facecolor('skyblue')
#ax.set_facecolor((1.0, 0.47, 0.42))
#plt.show()
if saveFile:
    plt.savefig("output/demandLoss_vs_fresh_policy="+policy+"_beta=_"+str(beta)\
                +"_cl="+str(cl)+"_n="+str(n)\
                +"_life="+str(m)+"_maxLead="+str(maxLead)\
                +"_leadAware="+str(leadAware)\
                +"_seed="+str(seed)+"_horizon="\
                +str(T)+"_numPol="+str(numSL)+"_numSim="+str(numSim)+"_"\
                +str(datetime.datetime.today())+".eps", format='eps', dpi=1000)
    plt.show()
else:
    plt.show()
    
########################################################
## Plot 4: the Pareto front of waste vs freshness
########################################################
for sc in range(numScenarios):
    plt.scatter(output[:,:,2,sc].sum(axis=1)/numSim, \
                output[:,:,4,sc].sum(axis=1)/numSim, \
                c = sColors[sc], s = 10)
                #c = 1-output[:,:,1,sc].sum(axis=1)/numSim, s = 10)
#plt.plot(output[:,2], output[:,3], '-')
#plt.xlim(0, 30)
#plt.ylim(0, 12)
#plt.title('TITLE', fontsize=20)
plt.xlabel('Average waste', fontsize=15)
plt.ylabel('Average freshness', fontsize=15)
plt.gray()
ax = plt.gca()
ax.set_facecolor('skyblue')
#ax.set_facecolor((1.0, 0.47, 0.42))
#plt.show()
if saveFile:
    plt.savefig("output/fresh_vs_waste_policy="+policy+"_beta=_"+str(beta)\
                +"_cl="+str(cl)+"_n="+str(n)\
                +"_life="+str(m)+"_maxLead="+str(maxLead)\
                +"_leadAware="+str(leadAware)+"_seed="+str(seed)+"_horizon="\
                +str(T)+"_numPol="+str(numSL)+"_numSim="+str(numSim)+"_"\
                +str(datetime.datetime.today())+".eps", format='eps', dpi=1000)
    plt.show()
else:
    plt.show()


########################################################
## Plot 1: the Pareto front of demand loss vs wastage
########################################################
for sc in range(numScenarios):
    plt.scatter(output[:,:,2,sc].sum(axis=1)/numSim, \
                output[:,:,3,sc].sum(axis=1)/numSim, \
                c = sColors[sc], s = 10)
#plt.plot(output[:,2], output[:,3], '-')
#plt.xlim(0, 30)
#plt.ylim(0, 12)
#plt.title('TITLE', fontsize=20)
plt.xlabel('Average waste', fontsize=15)
plt.ylabel('Average demand loss', fontsize=15)
plt.gray()
ax = plt.gca()
ax.set_facecolor('skyblue')
#ax.set_facecolor((1.0, 0.47, 0.42))
#plt.show()
if saveFile:
    plt.savefig("output/demandLoss_vs_waste_policy="+policy+"_beta=_"+str(beta)\
                +"_cl="+str(cl)+"_n="+str(n)\
                +"_life="+str(m)+"_maxLead="+str(maxLead)\
                +"_leadAware="+str(leadAware)+"_seed="+str(seed)+"_horizon="\
                +str(T)+"_numPol="+str(numSL)+"_numSim="+str(numSim)+"_"\
                +str(datetime.datetime.today())+".eps", format='eps', dpi=1000)
    plt.show()
else:
    plt.show()
    
############# Inv distribution ##########3


###############################################################################
############################# Main Simulation #################################
###############################################################################
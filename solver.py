import numpy as np
import networkx as nx
import instance as i
import itertools
import gurobipy as gb
import time



def FPTAS(ins: i.Instance, T: nx.DiGraph, epsilon = .5, bound = None):
    # rounding
    n = len([k[1] for k in T.nodes("u") if k[1] != None])
    P = max([k[1] for k in T.nodes("u") if k[1] != None])
    print(n*P)
    K = epsilon*P/n
    print("K ", K)
    # rounding
    for q in T.nodes():
        if T.nodes[q]['t'] == 'a':
            # print(q)
            T.nodes[q]["u_"] = np.ceil(T.nodes[q]['u']/K)
            # print("u = {} --> u_ = {}".format(T.nodes[q]["u"], T.nodes[q]["u_"]))
    print("max = " , max([k[1] for k in T.nodes("u_") if k[1] != None]))
    if bound == None:
        # VV = np.ceil(n*P/K)
        VV = np.floor(n*P/K)
    else:
        # VV = np.ceil(bound/K)
        VV = np.floor(bound / K)
    print(VV)
    L = np.arange(VV+1)
    # L = L[int(VV+1):0:-1] # reverse L for min objective

    D = {}
    q_count = {} # how many times visited during dfs
    for l in L:
        print("l = ", l)
        D[(q,0,l)] = np.infty


        for q in AND(ins,()):
            # print("q: {} | l: {}".format(q, l))
            # print(D)
            i = q_count.get((q,l),0)
            if T.nodes[q]['t'] == 'o' and i == 0:
                D[(q,i,l)] = np.infty
            elif len(q) == 1 and i == 0 and T.nodes[q]['t'] == 'a':
                if T.nodes[q]['u_'] <= l:
                    D[(q,0,l)] = T.nodes[q]['r']
                else:
                    D[(q,0,l)] = np.infty
            elif len(q) > 1 and i == 0 and T.nodes[q]['t'] == 'a':
                # TODO: Huge assumption! Observations are ordered numbers 0,1,2
                obs = q[-2:-1][0] # grand parents observation o
                if D[(q[:-2],max(obs - 1, 0), max(l - T.nodes[q]['u_'],0) ) ] <= ins.delta:
                    D[(q,0,l)] = D[(q[:-2], max(obs - 1, 0), max(l - T.nodes[q]['u_'], 0))]
                else:
                    D[(q,0,l)] = np.infty
            elif i>0 and T.nodes[q]['t'] == 'o': # R4
                # TODO: Another assumption
                act = i-1 # ith last action
                if ins.duration_model(q+(act,)) < ins.horizon:
                    D[(q,i,l)] =  min(D[(q,i-1,l)], D[(q+(act,), act, l)] )
                else:
                    D[(q,i,l)] =  min(D[(q,i-1,l)], D[(q+(act,), 0, l)] )

            elif i>0 and T.nodes[q]['t'] == 'a' and ins.duration_model(q) < ins.horizon: #R5
                # TODO: Another assumption
                act = i-1 # ith last action
                D[(q,i,l)] = D[(q+(act,), act, l)]

            q_count[(q,l)] = i + 1
    return D

# depth first search order traversal with rep
def AND(ins: i.Instance, p):
    for a in np.arange(len(ins.actions)):
        q = p + (a,) # (ins.action_to_string(a),)
        yield q
        if ins.duration_model(q) <  ins.horizon:
            for c in OR(ins, q):
                yield c
        yield p
def OR(ins: i.Instance, p):
    for o in np.arange(len(ins.observations)):
        q = p + (o,)
        yield q
        for c in AND(ins, q):
            yield c
        yield p

# def AND_OR_Traverse(ins: i.Instance, p):
#     yield p
#     for a in ins.actions:
#         q = p + (ins.action_to_string(a),)
        # q = p + (a,)
        # yield q
        # if ins.duration_model(q) < ins.horizon:
        #     for o in ins.observations:
        #         c = q + (o,)
        #         for _c in AND_OR_Traverse(ins, c):
        #             yield _c
        #         yield q
        #         yield c
        # else:
        #     yield p
    # yield p

# assume safe_at0 = 1
def preprocess(ins: i.Instance):
    T = nx.DiGraph()
    T.add_node((), b=ins.b0, rho=1, t='o')
    size_of_tree_ending_with_obs = sum([len(ins.actions)**i*len(ins.observations)**(i) for i in range(1,ins.horizon+1)]) + 1
    print("size of tree ending up with obs = ", size_of_tree_ending_with_obs)

    size_of_tree_ending_with_action = sum(
        [len(ins.actions) ** i * len(ins.observations) ** (i - 1) for i in range(1, ins.horizon + 1)]) + 1
    print("number of action vars = ", size_of_tree_ending_with_action - 1)
    size_of_and_or_tree = sum(
        [len(ins.actions) ** i * len(ins.observations) ** (i - 1) for i in range(1, ins.horizon + 1)]) \
                          + sum([len(ins.actions) ** i * len(ins.observations) ** i for i in range(1, ins.horizon)]) + 1
    print("size of AND-OR tree = ", size_of_and_or_tree)

    ILP_vars = []
    for q in AND_OR(ins, ()):
        if len(q) != 0:
            T.add_edge(q[:-1], q)
            parent = [i for i in T.predecessors(q)][0]
            # print(parent, "-->", q)
            # print('parent belief = ', T.nodes[parent]['b'])

            if len(q) % 2 == 0:  # observation
                # type
                T.nodes[q]['t'] = 'o'
                grand_parent = [i for i in T.predecessors(parent)][0]
                g_rho = T.nodes[grand_parent]['rho']
                b, p = _update_safe_belief(ins, safe_belief=T.nodes[parent]['b'], o=q[-1:][0])
                p_action = q[-2:-1][0]  # parent action

                safe_at = 1 - sum([b.get(s, 0) * ins.risk_model(s, p_action) for s in ins.risk_states])
                ## TODO require equation modification !!
                # g_b = T.nodes[grand_parent]['b']
                # safe_at
                #
                # = 1 - sum([g_b.get(s, 0) * ins.risk_model(s, p_action) for s in ins.risk_states])
                T.nodes[q]['p'] = p
                T.nodes[q]['rho'] = g_rho * p * safe_at
                # T.nodes[q]['safe_at'] = safe_at

            else:  # action
                T.nodes[q]['t'] = 'a'
                p_rho = T.nodes[parent]['rho']
                p_b = T.nodes[parent]['b']
                action = q[-1:][0]
                b = _update_safe_belief(ins, safe_belief=T.nodes[parent]['b'], a=action)
                T.nodes[q]['r'] = p_rho * sum([b.get(s, 0) * ins.risk_model(s, action) for s in ins.risk_states])
                T.nodes[q]['u'] = p_rho * sum([v * ins.reward_model(s, action) for s, v in p_b.items()])
                # T.nodes[q]['h'] = p_rho * sum([v*ins.reward_heuristic(s,action) for s, v in b.items()])
                T.nodes[q]['h'] = p_rho * (1-ins.delta) * sum([v*ins.reward_heuristic(s,action) for s, v in p_b.items()])
                ILP_vars.append(q)
            T.nodes[q]['b'] = b
            # print('node belief = ', T.nodes[q]['b'])

    print("# nodes = ", len(T))
    return T, ILP_vars


# depth first search
def AND_OR(ins: i.Instance, p):
    for a in np.arange(len(ins.actions)):
        # q = p + (ins.action_to_string(a),)
        q = p + (a,)
        yield q
        if ins.duration_model(q) < ins.horizon:
            for o in np.arange(len(ins.observations)):
                c = q + (o,)
                yield c
                for _c in AND_OR(ins, c):
                    yield _c

def ILP(ins: i.Instance, T: nx.DiGraph, var_idx: list, continuous=False ):
    t1 = time.time()
    m = gb.Model("ILP")
    m.setParam("OutputFlag", 0)
    x={}
    for q in var_idx:
        x[q] = m.addVar(vtype=gb.GRB.BINARY, name=str(q)) if not continuous else   m.addVar(vtype=gb.GRB.BINARY, name=str(q))
    # obj = gb.quicksum([x[q]*T.nodes[q]['u'] for q in var_idx])
    obj = gb.quicksum(
        [x[q] * T.nodes[q]['u'] if ins.duration_model(q) < ins.horizon else x[q] * T.nodes[q]['h'] for q in var_idx])

    m.setObjective(obj, gb.GRB.MINIMIZE) if ins.type == "min" else m.setObjective(obj, gb.GRB.MAXIMIZE)




    tree_c1 = gb.quicksum([x[(a,)] for a in np.arange(len(ins.actions))])
    m.addConstr(tree_c1 == 1, "tree_c1")

    for q in var_idx:
        if ins.duration_model(q) < ins.horizon: # replace with duration model
            for o in np.arange(len(ins.observations)):
                tree_c2 = gb.quicksum([x[q+(o,a)] for a in np.arange(len(ins.actions))])
                m.addConstr(tree_c2 == x[q], "tree_c{}".format(q))
                m.update()

    capacity_c = gb.quicksum([x[q]*T.nodes[q]['r'] for q in var_idx])
    m.addConstr(capacity_c <= ins.delta)

    m.update()
    m.optimize()

    return obj.getValue(),{k:v.x for k,v in x.items() if v.x > 0}, time.time()-t1


def p_ILP(m,ins: i.Instance, T: nx.DiGraph, expanded: list, frontier = [], continuous=True, risk = False, warm_start = {}):
    t1 = time.time()

    x={}

    if len(warm_start) != 0: # reuse old variables
        for q,v in warm_start.items():
            x[q] = v
        for q in set(frontier + expanded)-set([k for k in warm_start.keys()]):
            x[q] = m.addVar(vtype=gb.GRB.BINARY, name=str(q)) if not continuous else  m.addVar(vtype=gb.GRB.CONTINUOUS, name=str(q))
    else:
        for q in frontier + expanded:
            x[q] = m.addVar(vtype=gb.GRB.BINARY, name=str(q)) if not continuous else m.addVar(vtype=gb.GRB.CONTINUOUS,
                                                                                              name=str(q))
    if len(frontier) != 0 and len(expanded) != 0:
        obj = gb.quicksum([x[q] * T.nodes[q]['u'] if ins.duration_model(q)<ins.horizon else x[q] * T.nodes[q]['h'] for q in expanded]) + gb.quicksum([x[q] * (T.nodes[q]['h']) for q in frontier])
    elif len(frontier)!= 0 and len(expanded) == 0:
        obj = gb.quicksum([x[q] * T.nodes[q]['h'] for q in frontier])
    else:
        obj = gb.quicksum([x[q] * T.nodes[q]['u'] if ins.duration_model(q)<ins.horizon else x[q] * T.nodes[q]['h'] for q in expanded])

    m.setObjective(obj, gb.GRB.MINIMIZE) if ins.type == "min" else m.setObjective(obj, gb.GRB.MAXIMIZE)


    tree_c1 = gb.quicksum([x[(a,)] for a in np.arange(len(ins.actions))])
    m.addConstr(tree_c1 == 1, "tree_c1")

    for q in expanded:
        if ins.duration_model(q) < ins.horizon: # replace with duration model
            for o in np.arange(len(ins.observations)):
                tree_c2 = gb.quicksum([x[q+(o,a)] for a in np.arange(len(ins.actions))])
                # print("seq: ", q+(o,))
                m.addConstr(tree_c2 == x[q], "tree_c{}".format(q))
                m.update()

    # TODO update risk heuristic later
    if risk:
        if len(frontier) != 0 and len(expanded) != 0:
            capacity_c = gb.quicksum([x[q] * T.nodes[q]['r'] for q in expanded]) + gb.quicksum([x[q] * T.nodes[q]['r'] for q in frontier])
        elif len(frontier) != 0 and len(expanded) == 0:
            capacity_c = gb.quicksum([x[q] * T.nodes[q]['r'] for q in frontier])
        else:
            capacity_c = gb.quicksum([x[q] * T.nodes[q]['r'] for q in expanded])

        m.addConstr(capacity_c <= ins.delta)

    m.update()
    m.optimize()

    # return obj.getValue(),{k:v.x for k,v in x.items() if v.x > 0}, time.time()-t1 # only positive results

    # return obj.getValue(),{k:v.X for k,v in x.items()}, time.time()-t1
    return obj.getValue(),x, time.time()-t1

def heuristic_search(ins: i.Instance, continuous=False, risk=True ):
    total_num_action_vars = sum(
        [len(ins.actions) ** i * len(ins.observations) ** (i - 1) for i in range(1, ins.horizon + 1)]) + 1
    # print("number of action vars = ", total_num_action_vars - 1)
    size_of_and_or_tree = sum(
        [len(ins.actions) ** i * len(ins.observations) ** (i - 1) for i in range(1, ins.horizon + 1)]) \
                          + sum([len(ins.actions) ** i * len(ins.observations) ** i for i in range(1, ins.horizon)]) + 1
    # print("size of AND-OR tree = ", size_of_and_or_tree)

    m = gb.Model("p_ILP")
    m.setParam("OutputFlag", 0)

    t1 = time.clock()

    T = nx.DiGraph()
    T.add_node((), b=ins.b0, rho=1)
    expanded = []
    frontier = [(a,) for a in np.arange(len(ins.actions))]
    for q in frontier:
        a = q[0]
        T.add_edge((), q)
        b = _update_safe_belief(ins, safe_belief=T.nodes[()]['b'], a=a)
        p_b = T.nodes[()]['b']
        T.nodes[q]['r'] = sum([b.get(s, 0) * ins.risk_model(s, a) for s in ins.risk_states])
        T.nodes[q]['u'] = sum([v * ins.reward_model(s, a) for s, v in p_b.items()])
        # T.nodes[q]['h'] = sum(
        #     [v * ins.reward_model(s, a) + v * ins.reward_heuristic(s, a) for s, v in b.items()])
        T.nodes[q]['h'] = (1 - ins.delta) * sum([v * ins.reward_heuristic(s, a) for s, v in p_b.items()])
        T.nodes[q]['b'] = b


    loop_count = 0
    # main loop
    sol = {}
    while True:
        # print("iteration {}".format(loop_count))
        # print("frontier size     = {0:7d}".format(len(frontier)), " ({}%)".format(len(frontier)/total_num_action_vars * 100))
        # print("expanded size     = {0:7d}".format(len(expanded)), " ({}%)".format(len(expanded)/total_num_action_vars * 100))
        # print("Total Exploration = {0:7d}".format(len(frontier)+len(expanded)), "({}%)".format((len(frontier)+len(expanded))/total_num_action_vars * 100))
        loop_count+=1

        obj, sol, _ = p_ILP(m,ins,T, expanded=expanded, frontier = frontier, continuous=continuous, risk=risk, warm_start=sol)
        new = [q for q in frontier if sol[q].x > 0]
        expanded = expanded + new
        frontier = list(set(frontier) - set(new)) # + removed_roots
        for n in new:
            if ins.duration_model(n) < ins.horizon:
                for o in np.arange(len(ins.observations)):
                    q = n+(o,)
                    parent = n
                    T.add_edge(parent, q)
                    grand_parent = [i for i in T.predecessors(parent)][0]
                    g_rho = T.nodes[grand_parent]['rho']
                    b, p = _update_safe_belief(ins, safe_belief=T.nodes[parent]['b'], o=o)
                    p_action = n[-1:][0]  # parent action
                    safe_at = 1 - sum([b.get(s, 0) * ins.risk_model(s, p_action) for s in ins.risk_states])
                    ## TODO require equation modification !!
                    # g_b = T.nodes[grand_parent]['b']
                    # safe_at = 1 - sum([g_b.get(s, 0) * ins.risk_model(s, p_action) for s in ins.risk_states])
                    T.nodes[q]['p'] = p
                    T.nodes[q]['rho'] = g_rho * p * safe_at
                    T.nodes[q]['b'] = b

                    for action in np.arange(len(ins.actions)):
                        q = n+(o,action)
                        parent = q[:-1]
                        p_b = T.nodes[parent]['b']
                        T.add_edge(parent, q)
                        frontier.append(q)
                        p_rho = T.nodes[parent]['rho']
                        b = _update_safe_belief(ins, safe_belief=T.nodes[parent]['b'], a=action)
                        T.nodes[q]['r'] = p_rho * sum([b.get(s, 0) * ins.risk_model(s, action) for s in ins.risk_states])
                        T.nodes[q]['u'] = p_rho * sum([v * ins.reward_model(s, action) for s, v in p_b.items()])
                        # T.nodes[q]['h'] = p_rho * sum(
                        #     [ v * ins.reward_heuristic(s, action) for s, v in p_b.items()])
                        T.nodes[q]['h'] = p_rho * (1 - ins.delta) * sum(
                            [v * ins.reward_heuristic(s, action) for s, v in p_b.items()])
                        T.nodes[q]['b'] = b



        if len(new) == 0:
            break

    print("~~~~~~~~~~~~~~~~~~~~🔥 SUM 🔥~~~~~~~~~~~~~~~~~~~~")
    print("# iteration {}".format(loop_count))
    print("frontier size     = {0:7d}".format(len(frontier)), " ({}%)".format(len(frontier)/total_num_action_vars * 100))
    print("expanded size     = {0:7d}".format(len(expanded)), " ({}%)".format(len(expanded)/total_num_action_vars * 100))
    print("Total Exploration = {0:7d}".format(len(frontier)+len(expanded)), " ({}%)".format((len(frontier)+len(expanded))/total_num_action_vars * 100))
    t2 = time.clock() - t1
    print("Time              = {0:7.3f}".format(t2))
    print("~~~~~~~~~~~~~~~~~~~~🔥~~~~~🔥~~~~~~~~~~~~~~~~~~~~")

    return obj, {k:v.x for k,v in sol.items() if v.x>0}, t2



def _next_belief_states(ins: i.Instance, states, a):
    # maybe observation can optimize it a bit
    f = []
    for s in states:
        f = f + [s_ for s_ in ins.trans_model(s, a).keys()]
    f = set(f)
    f.union(states)
    return f


def _update_belief(ins: i.Instance, belief, a=None, o=None,normalize=True):
    if a == None and o == None:
        return belief
    prior = {}
    posterior = {}
    prob_o = 0

    if a == None:
        for s, v in belief.items():
            posterior[s] = ins.obs_model(s, a)[o] * v
            prob_o += posterior[s]
        if normalize:
            safe_posterior = {k: v / prob_o for k, v in posterior.items() if v != 0}
        return safe_posterior, prob_o

    next_belief_states = _next_belief_states(ins, belief.keys(), a)

    # print("frontier: ", next_belief_states)
    # print("safe frontier: ", next_safe_belief_states)

    for s in next_belief_states:
        S = ins.states_reachable_to[s]
        val = sum([belief.get(s_, 0) * ins.trans_model(s_, a).get(s, 0) for s_ in S])
        if val != 0:
            prior[s] = val
            if o != None:
                posterior[s] = ins.obs_model(s, a)[o] * prior[s]
                prob_o += posterior[s]

    if o == None:
        return prior

    # normalize
    if normalize:
        posterior = {k: v / prob_o for k, v in posterior.items()}
    return posterior, prob_o


def _update_safe_belief(ins: i.Instance, safe_belief, a=None, o=None, normalize=True):
    if a == None and o == None:
        return safe_belief
    safe_prior = {}
    safe_posterior = {}
    safe_prob_o = 0

    if a == None:
        for s, v in safe_belief.items():
            safe_posterior[s] = ins.obs_model(s, a)[o] * v
            safe_prob_o += safe_posterior[s]
        if normalize:
            safe_posterior = {k: v / safe_prob_o for k, v in safe_posterior.items() if v != 0}
        return safe_posterior, safe_prob_o

    next_safe_belief_states = _next_belief_states(ins, safe_belief.keys(), a)
    # next_safe_belief_states = ins.states
    safe_at = 1 - sum([v * ins.risk_model(s, a) for s, v in safe_belief.items()])
    for s in next_safe_belief_states:
        # S_ = ins.states_reachable_to[s] - set(ins.risk_states)
        # val_ = sum([safe_belief.get(s_, 0) * ins.trans_model(s_, a).get(s, 0) for s_ in S_]) / safe_at
        val_ = sum([safe_belief.get(s_, 0) * ins.trans_model(s_, a).get(s, 0) * (1 - ins.risk_model(s_, a)) for s_ in
                    ins.states_reachable_to[s]]) / safe_at
        if val_ != 0:
            safe_prior[s] = val_
            if o != None:
                safe_posterior[s] = ins.obs_model(s, a)[o] * safe_prior[s]
                safe_prob_o += safe_posterior[s]

    if o == None:
        return safe_prior

    if normalize:
        safe_posterior = {k: v / safe_prob_o for k, v in safe_posterior.items()}
    return safe_posterior, safe_prob_o

##################################################
#### functions used debugging and experimentations
##################################################


# returns <a_1,o_1,a_2,o_2,...,a_t,o_t>
def BFS(ins: i.Instance):
    # yield ()
    for i in np.arange(0, ins.horizon + 1):
        for seq in itertools.product(np.arange(len(ins.actions)), np.arange(len(ins.observations)), repeat=i):  # depth first search
            yield seq


# returns <a_1,o_1,a_2,o_2,...,o_t,a_t+1>
def BFS_(ins: i.Instance):
    for a in np.arange(len(ins.actions)):
        yield (a,)
    for a in np.arange(len(ins.actions)):
        for i in np.arange(1, ins.horizon):
            for seq in itertools.product(np.arange(len(ins.observations)), np.arange(len(ins.actions)), repeat=i):  # depth first search
                q = (a,) + seq
                yield q

# assume a_seq and o_seq have the same size
def history(ins: i.Instance, current_belief=None, current_safe_belief=None, a_seq=[0], o_seq=None):
    if current_belief == None:
        current_belief = ins.b0
    if current_safe_belief == None:
        current_safe_belief = ins.b0

    for i in range(len(a_seq)):
        if o_seq == None:
            b = _update_belief(ins, belief=current_belief, a=a_seq[i])
            b_ = _update_safe_belief(ins, safe_belief=current_safe_belief, a=a_seq[i])
        else:
            b, p = _update_belief(ins, belief=current_belief, a=a_seq[i], o=o_seq[i])
            b_, p_ = _update_safe_belief(ins, safe_belief=current_safe_belief, a=a_seq[i], o=o_seq[i])
            print("p_ = ", p_)
            print("Action {} Observation {}".format(ins.action_to_string(a_seq[i]), o_seq[i]))
        current_belief = b
        current_safe_belief = b_
        # print("posterior      = ", current_belief)
        # print("p = ", p)
        print("safe posterior = ", current_safe_belief)
        print()

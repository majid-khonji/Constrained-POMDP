import numpy as np
import matplotlib.pyplot as plt
import scipy.special as s

class car:
    def __init__(self, theta=np.pi/4, W=2, L=4, bx=3, by=3, cov=np.zeros((2,2)), name = "", tolerance=0.0000000001):
        self.theta = theta
        self.W = W
        self.L = L
        self.loc = (bx,by)
        self.loc_noise = (0,0)
        self.cov = cov
        self.name = name

        self.tolerance =tolerance # used when testing feasibility when sampling

        a1 = -np.tan(theta); b1 = 1; c1 = np.tan(theta) * bx - by - W / 2 * np.abs(1 / np.cos(theta))
        l_1 = lambda x, y: a1* x + b1*y  + c1

        a2 = np.tan(theta); b2 = -1; c2 = - np.tan(theta) * bx + by - W / 2 * np.abs(1 / np.cos(theta))
        l_2 = lambda x, y: a2 * x + b2 * y + c2

        a3 = - 1/np.tan(theta); b3 = -1; c3= 1 / np.tan(theta) * bx + by - L / 2 * np.abs(1 / np.sin(theta))
        l_3 = lambda x, y: a3 * x + b3*y + c3

        a4 = 1/np.tan(theta); b4 = 1; c4=- 1 / np.tan(theta) * bx - by - L/ 2 * np.abs(1 / np.sin(theta))
        l_4 = lambda x, y: a4 * x + b4*y  + c4

        y_1 = lambda x: -l_1(x,0)/b1
        y_2 = lambda x: -l_2(x,0)/b2
        y_3 = lambda x: -l_3(x,0)/b3
        y_4 = lambda x: -l_4(x,0)/b4
        # might be useful for visualization: y = m x + c or ax + by + c = 0
        self.lines = [y_1, y_2, y_3, y_4]

        # gamma_k std
        std_gamma1 = np.sqrt( np.tan(self.theta) ** 2 * self.cov[0][0] + self.cov[1][1] - 2 * np.tan(self.theta) * self.cov[0][1]  )
        std_gamma2 = std_gamma1
        std_gamma3 = np.sqrt( 1/np.tan(self.theta) ** 2 * self.cov[0][0] + self.cov[1][1] + 2 * 1/np.tan(self.theta) * self.cov[0][1]  )
        std_gamma4 = std_gamma3

        self.a = [a1,a2,a3,a4]
        self.b = [b1,b2,b3,b4]
        self.c = [c1,c2,c3,c4]
        self.d = [-np.tan(theta), np.tan(theta), -1/np.tan(theta), 1/np.tan(theta)]
        # self.d = [np.tan(theta), -np.tan(theta), 1/np.tan(theta), -1/np.tan(theta)]
        self.f = [1,-1,-1,1]
        self.sampled_gamma = [0,0,0,0]
        self.std_gamma = [std_gamma1, std_gamma2, std_gamma3, std_gamma4]
        self.line_functions = [l_1, l_2, l_3, l_4]

        # vehicle corners
        self.corners = [self._line_intersect(0, 2), self._line_intersect(0, 3), self._line_intersect(1, 2), self._line_intersect(1, 3)]





    # this function adds a gaussian noise. updates corners
    def sample(self):
        omega = np.random.multivariate_normal([0, 0], self.cov)
        self.loc_noise = (omega[0], omega[1])
        gamma_1 = - np.tan(self.theta) * omega[0] + omega[1]
        gamma_2 =  np.tan(self.theta) * omega[0] - omega[1]
        gamma_3 = - 1/np.tan(self.theta) * omega[0] -  omega[1]
        gamma_4 =  1/np.tan(self.theta) * omega[0] + omega[1]
        self.sampled_gamma = [gamma_1, gamma_2, gamma_3, gamma_4]
        self.corners = [self._line_intersect(0, 2), self._line_intersect(0, 3), self._line_intersect(1, 2), self._line_intersect(1, 3)]
        # y_1 = lambda x: -(self.line_functions[0](x,0) + self.gamma_1)/self.b[0]
        # y_2 = lambda x: -(self.line_functions[1](x,0) + self.gamma_2)/self.b[1]
        # y_3 = lambda x: -(self.line_functions[2](x,0) + self.gamma_3)/self.b[2]
        # y_4 = lambda x: -(self.line_functions[3](x,0) + self.gamma_4)/self.b[3]
        # self.lines = [y_1,y_2, y_3, y_4]


    def _line_intersect(self, l1, l2):
        # x = - (self.b[l1]*self.c[l2] - self.c[l1]*self.b[l2])/(self.b[l1] * self.a[l2] - self.a[l1]*self.b[l2])
        # y = (self.a[l1]*self.c[l2] - self.c[l1]*self.a[l2])/(self.b[l1] * self.a[l2] - self.a[l1]*self.b[l2])

        x = - (self.b[l1]*self.c[l2] - self.c[l1]*self.b[l2] - self.b[l1]*self.sampled_gamma[l2] + self.sampled_gamma[l1]*self.b[l2])/(self.b[l1] * self.a[l2] - self.a[l1]*self.b[l2])
        y = (self.a[l1]*self.c[l2] - self.c[l1]*self.a[l2] - self.a[l1]*self.sampled_gamma[l2] + self.sampled_gamma[l1]*self.a[l2])/(self.b[l1] * self.a[l2] - self.a[l1]*self.b[l2])
        return (x,y)

    # checks points feasibility within
    def check_feasibility(self, p, tolerance=None):
        if tolerance != None:
            self.tolerance = tolerance

        for i in np.arange(4):
            l = self.line_functions[i]
            if l(p[0],p[1]) > self.sampled_gamma[i] +self.tolerance:
                return False
        return True


    def plot(self, show_line_labels = False, alpha=1, color="red"):
        plt.plot([self.loc[0]], [self.loc[1]], marker='+', markersize=6, color=color, alpha=alpha)
        plt.text(self.loc[0]+.1, self.loc[1]+.1, self.name, alpha=alpha)

        t0 = self.corners[0]
        t1 = self.corners[1]
        plt.plot([t0[0], t1[0]], [t0[1],t1[1]], label="$\ell_0 = 0$",alpha=alpha)
        t0 = self.corners[2]
        t1 = self.corners[3]
        plt.plot([t0[0], t1[0]], [t0[1],t1[1]], label="$\ell_1 = 0$", alpha=alpha)
        t0 = self.corners[0]
        t1 = self.corners[2]
        plt.plot([t0[0], t1[0]], [t0[1],t1[1]], label="$\ell_2=0$", alpha=alpha)
        t0 = self.corners[1]
        t1 = self.corners[3]
        plt.plot([t0[0], t1[0]], [t0[1],t1[1]], label="$\ell_3=0$", alpha=alpha)

        plt.axis('equal')
        if show_line_labels:
            plt.legend()

        for p in self.corners:
            plt.plot([p[0]], [p[1]], marker='o', markersize=5, color=color, alpha=alpha)

    # also shows the shift
    def plot_sample(self, show_line_labels = False, alpha=1, color="red"):
        plt.plot([self.loc[0]], [self.loc[1]], marker='+', markersize=6, color=color, alpha=alpha)
        plt.plot([self.loc[0]+self.loc_noise[0]], [self.loc[1]+self.loc_noise[1]], marker='+', markersize=15, color=color, alpha=alpha)

        plt.text(self.loc[0]+self.loc_noise[0] +.1, self.loc[1]+self.loc_noise[1]+.1, self.name, alpha=alpha)

        # shift line
        plt.plot([self.loc[0], self.loc[0]+ self.loc_noise[0]], [self.loc[1], self.loc[1]+ self.loc_noise[1]], alpha=alpha, linestyle=':', color=color)

        t0 = self.corners[0]
        t1 = self.corners[1]
        plt.plot([t0[0], t1[0]], [t0[1],t1[1]], label="$\ell_0 = 0$",alpha=alpha, color=color)
        t0 = self.corners[2]
        t1 = self.corners[3]
        plt.plot([t0[0], t1[0]], [t0[1],t1[1]], label="$\ell_1 = 0$", alpha=alpha, color=color)
        t0 = self.corners[0]
        t1 = self.corners[2]
        plt.plot([t0[0], t1[0]], [t0[1],t1[1]], label="$\ell_2=0$", alpha=alpha, color=color)
        t0 = self.corners[1]
        t1 = self.corners[3]
        plt.plot([t0[0], t1[0]], [t0[1],t1[1]], label="$\ell_3=0$", alpha=alpha, color=color)

        plt.axis('equal')
        if show_line_labels:
            plt.legend()

        for p in self.corners:
            plt.plot([p[0]], [p[1]], marker='o', markersize=5, color=color, alpha=alpha)

def _line_intersect(l_k, l_k_, sampled_gamma_k=0, sampled_gamma_k_=0):
    a_k = l_k[0]; b_k=l_k[1]; c_k=l_k[2]
    a_k_ = l_k_[0]; b_k_=l_k_[1]; c_k_=l_k_[2]
    if (a_k != a_k_) or (b_k != b_k_):
        x = - (b_k * c_k_ - c_k * b_k_ - b_k * sampled_gamma_k_ + sampled_gamma_k * b_k_) / (b_k * a_k_ - a_k * b_k_)
        y = (a_k * c_k_ - c_k * a_k_ - a_k * sampled_gamma_k_ + sampled_gamma_k * a_k_) / (b_k * a_k_ - a_k * b_k_)
        return (x,y)
    return None

def _candidate_collision_points(c1,c2, idx=None):
    C = []
    if idx != None:
        for x in idx:
            k = x[0][1]  # kth line of for i
            k_ = x[1][1]  # k_th line for j
            l1 = [c1.a[k], c1.b[k], c1.c[k]]
            l2 = [c2.a[k_], c2.b[k_], c2.c[k_]]
            p = _line_intersect(l1, l2, c1.sampled_gamma[k], c2.sampled_gamma[k_])
            if p != None:
                C.append(p)
    else:
        idx = []
        for i in np.arange(4):
            for j in np.arange(4):
                l1 = [c1.a[i], c1.b[i], c1.c[i]]
                l2 = [c2.a[j], c2.b[j], c2.c[j]]
                p = _line_intersect(l1,l2,c1.sampled_gamma[i], c2.sampled_gamma[j])
                if p != None:
                    C.append(p)
                    idx.append([(0,i), (1,j)])

    return C, idx


def collision_points(c1,  c2):
    cand,_  = _candidate_collision_points(c1,c2)
    return [c for c in cand if c1.check_feasibility(c) and c2.check_feasibility(c)]

def select_collision_points(c1,  c2, idx, violate_lines): # idx line intersection that violate violate_lines
    cand,_  = _candidate_collision_points(c1,c2, idx = idx)
    cars = [c1,c2]
    C = []
    for c in cand:
        for l in violate_lines:
            m = l[0]
            p = l[1]
            line = cars[m].line_functions[p]
            # print("select_col_points: l>0?", line(c[0], c[1]) - cars[m].sampled_gamma[p])
            if line(c[0], c[1]) > cars[m].sampled_gamma[p] + cars[m].tolerance:
                C.append(c)
    return C

def test_collision(c1, c2, idx=None, violate_lines = None):
    cars = [c1,c2]
    cand, _ = _candidate_collision_points(c1,c2, idx = idx)
    if violate_lines != None:
        for c in cand:
            for l in violate_lines:
                m = l[0]
                p = l[1]
                line = cars[m].line_functions[p]
                if line(c[0], c[1]) > cars[m].sampled_gamma[p] + cars[m].tolerance:
                    return True
    else:
        for c in cand:
            if c1.check_feasibility(c) and c2.check_feasibility(c):
                return True
    return False
def prob_collision(c1, c2, sample_size = 100, idx = None, violate_lines=None):
    col_count = 0
    for i in range(sample_size):
        c1.sample()
        c2.sample()
        if test_collision(c1, c2, idx, violate_lines):
            col_count += 1
    if violate_lines != None:
        return 1 -col_count/sample_size
    else:
        return col_count/sample_size

def prob_collision_upper_bound(c1, c2):
    car = [c1, c2]
    C, idx = _candidate_collision_points(c1, c2) # no need to iterate here
    ###############################################3
    ###############################################3
    ###############################################3
    ###############################################3
    # idx = [[(0,1),(1,0)], [(0,0), (1,1)], [(0,2), (1,3)], [(0,3),(0,2)]] #########################3
    # idx = [[(0,1),(1,0)]]
    ###############################################3
    ###############################################3
    ###############################################3
    cov = car[0].cov
    cov_ = car[1].cov

    sum_prob = 0
    total_sum = 0
    c = 0
    for x in range(len(idx)):
        print(idx[x])
        k = idx[x][0][1] # kth line of for i
        k_ = idx[x][1][1] # k_th line for j

        # iterating over list L,
        max_of_cars = 0
        pr = [[],[]]
        for m in [0,1]:
            sum_per_car = 0
            k__ = idx[x][m][1]  # k__th line for m
            for p in np.arange(4):
                a_p = car[m].a[p]
                b_p = car[m].b[p]
                a_v = car[m].a[k__]
                b_v = car[m].b[k__]
                parallel = ( a_p == a_v and b_p == b_v ) or ( a_p == -a_v and b_p == -b_v )
                same = p == idx[x][m][1]
                if not same and not parallel:
                # if (m, p) in idx[x]:
                    print("car %d: line %d"%(m,p))
                    c += 1
                    a_p = car[m].a[p]
                    b_p = car[m].b[p]
                    c_p = car[m].c[p]
                    d_p = car[m].d[p]
                    f_p = car[m].f[p]

                    a_k = car[0].a[k]
                    b_k = car[0].b[k]
                    c_k = car[0].c[k]
                    d_k = car[0].d[k]
                    f_k = car[0].f[k]

                    a_k_ = car[1].a[k_]
                    b_k_= car[1].b[k_]
                    c_k_= car[1].c[k_]
                    d_k_ = car[1].d[k_]
                    f_k_ = car[1].f[k_]

                    A = (a_p*b_k_ - b_p*a_k_)/ (b_k*a_k_ - a_k *b_k_) # for w^i
                    B = (a_k*b_p - a_p*b_k)/(b_k*a_k_ - a_k*b_k_) # for w^j
                    if m == 0: # add d_k, f_k to r
                        r = np.array([ A*d_k + d_p, A*f_k+ f_p])
                        _r = np.array([ B*d_k_, B*f_k_])
                    elif m == 1:
                        r = np.array([ A*d_k, A*f_k])
                        _r = np.array([B * d_k_ + d_p, B * f_k_ + f_p])
                    Psi = r.transpose().dot(cov).dot(r) + _r.transpose().dot(cov_).dot(_r)
                    C_ = c_p - (( b_p*(c_k*a_k_- a_k * c_k_ ) + a_p*(b_k * c_k_ - c_k*b_k_) )/ (b_k*a_k_ - a_k*b_k_)  )
                    val = 0.5 * (1+ s.erf(C_/(np.sqrt(Psi)*np.sqrt(2))) )
                    sum_per_car += val
                    pr[m].append(val)
                    print("The value is  ", val)
                    print("____")

                    # nice debugging trick
                    # omega_i = np.array(c0.loc_noise)
                    # omega_j = np.array(c1.loc_noise)
                    # sam_R = r.dot(omega_i) + _r.dot(omega_j)
                    # print("R = ", sam_R)
                    # print("C_ = ", C_)
                    # print("l>0? ", C_ - sam_R )
                    # print("omega_i = ", omega_i)
                    # print("omega_j = ", omega_j)


            print("sum per car = ", sum_per_car)
            print("diff = ", 1-sum_per_car)

            if sum_per_car > max_of_cars:
                max_of_cars = sum_per_car
            print("++++++++++++++")
        # print("max sum prob = ", max_of_cars)
        # print("1-max sum  = ",1- max_of_cars)
        print("###################")
        print("point c max prob = ", max(pr[0]) + max(pr[1]))
        print("point c bound prob = ",max_of_cars)
        print("###################")
        # sum_prob += max_prob
        sum_prob += max_of_cars
        # sum_prob += max(pr[0]) + max(pr[1])
        # print(pr)

    prob_bound = len(idx) - sum_prob
    print("Paper bound:  ", prob_bound)

    print("# terms = ", c)
if __name__ == '__main__':
    pass

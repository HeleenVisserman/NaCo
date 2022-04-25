import math
from scipy import stats
from matplotlib import pyplot as plt

size = 10 #number of weak classifiers
p_weak = 0.6


def calc_correct_decision(ps, w): # 4
    res = 0

    r1 = max(math.ceil((size+1)/2 - w), 0)
    r2 = size//2+1

    print("r1", r1)
    print("r2", r2)

    # Caclulate for p(strong = correct)
    for value in range(r1, size + 1):
        # print(value)
        res += 0.8 * stats.binom.pmf(value, size, ps)

    
    # Calculate for p(strong = wrong)
    for value in range(r2, size + 1):
        # print(value)
        res += 0.2 * stats.binom.pmf(value, size, ps)

    return res

def plot_weights():
    # weights = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5]
    weights = [w for w in range(8)]
    results = []
    for w in weights:
        results.append(calc_correct_decision(p_weak, w))
    
    plt.plot(weights, results)
    plt.xlabel("weight")
    plt.ylabel("probability of correct decision")
    # plt.legend()
    plt.show()

def compute_ada_weight(error):
    return math.log((1.-error)/error)

def plot_ada_weights():
    probs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    results = []
    for p in probs:
        results.append(compute_ada_weight(p))
    plt.plot(probs, results)
    plt.xlabel("Error rate")
    plt.ylabel("AdaBoost weight")
    plt.show()

# =============== 3 b ==================
# print(calc_correct_decision(p_weak, 1))
# =============== 3 c ==================
# plot_weights()
# =============== 3 d ==================
# plot_ada_weights()
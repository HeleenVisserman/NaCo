import math
from scipy import stats
from matplotlib import pyplot as plt


def calc_correct_decision(size, ps):
    res = 0
    for value in range(size // 2 + 1, size+1):
        # print(value)
        res += stats.binom.pmf(value, size, ps)
    return res
    

def plot_2c():
    jury_sizes = [i for i in range(1,16)]
    competences = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    result = []
    for c in competences:
        res = []
        for s in jury_sizes:
            res.append(calc_correct_decision(s, c))   
        result.append(res)
    
    for res in range(len(result)):
        print(result[res])
        plt.plot(jury_sizes, result[res], label = f"p={competences[res]}")
        
    plt.xlabel("Jury Size")
    plt.ylabel("Probability of Correct Decision")
    plt.legend()
    plt.show()


def plot_2d():
    jury_sizes = [i for i in range(19,50)]
    competences = [0.6]
    result = []
    for c in competences:
        res = []
        for s in jury_sizes:
            res.append(calc_correct_decision(s, c))   
        result.append(res)
    
    for res in range(len(result)):
        print(result[res])
        plt.plot(jury_sizes, result[res], label = f"p={competences[res]}")

    plt.plot(jury_sizes, [0.896]*len(jury_sizes), label = "doctors")    
    plt.xlabel("Jury Size")
    plt.ylabel("Probability of Correct Decision")
    plt.legend()
    plt.show()


# =============== 2 b ==================
# print(calc_correct_decision(19, 0.6))

# =============== 2 c ==================
# plot_2c()

# =============== 2 d ==================
# print(calc_correct_decision(1, 0.85))
# print(calc_correct_decision(3, 0.8))
# print(calc_correct_decision(19, 0.6))
# plot_2d()
# print(calc_correct_decision(39,0.6))

# =============== 3 a ==================
print(calc_correct_decision(10, 0.6))
import click
import Generator
import matplotlib.pyplot as plt
import collections
import pandas
import numpy as np
from sklearn.preprocessing import normalize
from scipy import stats

class Simulation:
    def __init__(self, p, q, n, T) -> None:
        self.p = p
        self.q = q
        self.n = n
        self.T = T
        self.generator = Generator.EdgeMarkovianGenerator(p, q, n, T)

    def simulate(self):
        self.generator.simulate()

    def getResult(self):
        return list(self.generator.getDensityMap())

def simulate(p, q, n, T):
    simu = Simulation (p, q, n, T)
    simu.simulate()
    return simu.getResult()

def truncate(x, k):
    power = 10 ** k
    return int(x * power) / power

def truncatek(k):
    return lambda x : truncate(x, k)

@click.command()
@click.argument("p", type=float)
@click.argument("q", type=float)
@click.argument("n", type=int)
@click.argument("t", type=int)
@click.argument("precision", type=int)
@click.argument("mode", type=str)
def main(p, q, n, t, precision, mode):
    simuTruncated = list(map(truncatek(precision), simulate(p, q, n, t)))

    if mode == "occurences":
        occurences = collections.OrderedDict(sorted(collections.Counter(simuTruncated).items()))
        occurences_x = list(occurences.keys())
        occurences_y = list(occurences.values())

        # The estimated limit if one exists
        if (p + q != 2):
            estimatedDensity = (1 - q) / (2 - p - q)
            plt.plot([estimatedDensity]*2, [min(occurences_y), max(occurences_y)], "--", label="Estimated average", color="red")
        
        # Plot it first using matplotlib
        plt.plot(occurences_x, occurences_y, label="Distribution density", color="blue")
        plt.xlabel("density")
        plt.ylabel("Occurences of density")
        plt.title(f"Distribution of densitiy on a whole simulation\nHere p={p}, q={q} and T={t}\tf'(m)={p + q - 1}")
        plt.legend(loc="best")
        plt.savefig(f"occurences_of_density_{p}_{q}_{n}_{t}.png")

        # Then save the data in a csv
        df = pandas.DataFrame({"Density" : occurences_x, "Occurences" : occurences_y})
        df.to_csv(f"occurences_of_density_{p}_{q}_{n}_{t}.csv")

    elif mode == "temporal":

        # Plot it first using matplotlib
        plt.plot(list(range(t+1)), simuTruncated, label="Densitiy evolution")
        # The estimated limit if one exists
        if (p + q != 2):
            estimatedDensity = (1 - q) / (2 - p - q)
            plt.plot([0, t], [estimatedDensity]*2, "--", label="Estimated average", color="red")
        plt.xlabel("t")
        plt.ylabel("density")
        plt.ylim((0, 1))
        plt.title(f"Evolution through of densitiy on a whole simulation\nHere p={p}, q={q} and T={t}\tf'(m)={p + q - 1}")
        plt.legend(loc="best")
        plt.savefig(f"evolution_of_density_{p}_{q}_{n}_{t}.png")

        # Then save it in a csv
        df = pandas.DataFrame({"Time" : list(range(t+1)), "Density" : simuTruncated})
        df.to_csv(f"evolution_of_density_{p}_{q}_{n}_{t}.csv")

    elif mode == "gaussianComparison":
        occurences = collections.OrderedDict(sorted(collections.Counter(simuTruncated).items()))
        occurences_x = list(occurences.keys())
        occurences_y = list(occurences.values())
        occurences_y = np.array(occurences_y)
        #occurences_y = occurences_y / np.linalg.norm(occurences_y)

        # The estimated limit if one exists
        #if (p + q != 2):
        #    estimatedDensity = (1 - q) / (2 - p - q)
        #    plt.plot([estimatedDensity]*2, [min(occurences_y), max(occurences_y)], "--", label="Estimated average", color="red")

        # Save the data in a csv
        df = pandas.DataFrame({"Density" : occurences_x, "Occurences" : occurences_y})
        df.to_csv(f"occurences_of_density_{p}_{q}_{n}_{t}.csv")

        # Plot a gaussian bell curve
        df = pandas.DataFrame({"Time" : list(range(t+1)), "Density" : simuTruncated})
        meanDensity = df.mean()["Density"]
        stdDensity  = df.std()["Density"]
        print(f"{meanDensity}, {stdDensity}")
        m, M        = df.min()["Density"], df.max()["Density"]
        rangeDensity = np.linspace(m, M, 100)
        gaussianApprox = stats.norm.pdf(rangeDensity, meanDensity, stdDensity)#1/(stdDensity * np.sqrt(2 * np.pi)) * np.exp( - (rangeDensity - meanDensity)**2 / (2 * (stdDensity**2)))
        plt.plot(rangeDensity, gaussianApprox, "--", color="black", label="Gaussian approximation")        
        
        # Plot it first using matplotlib
        plt.hist(simuTruncated, bins=100, density=True, label="Distribution density", color="blue")
        plt.xlabel("density")
        plt.ylabel("Occurences of density")
        plt.title(f"Distribution of densitiy on a whole simulation\nHere p={p}, q={q} and T={t}\tf'(m)={p + q - 1}")
        plt.legend(loc="best")
        plt.savefig(f"comparison_occurences_of_density_{p}_{q}_{n}_{t}.png")



if __name__ == "__main__":
    main()
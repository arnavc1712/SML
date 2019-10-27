from classes.kmeans import KMeans
import os
import matplotlib.pyplot as plt
cur_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cur_dir, "./AllSamples.mat")



def compute_all_objective_costs(range_clusters,strategy,random_seed):
    all_costs = []
    ## Computing all objective functions for clusters from (2,range_clusters) and certain strategy
    print("\n\n Objective Function \n")
    print("Strategy {}----- Random Seed {}".format(str(strategy),str(random_seed)))
    for i in range(2,range_clusters+1):
        k = KMeans(i,data_path,random_seed=random_seed,strategy=strategy)
        k.kmeans()
        # k.plot_points()
        cost = k.objective_function()
        all_costs.append(cost)
        print("For {} Clusters: {}".format(str(i),str(round(cost,5))))
    return all_costs


def plot_objective_function(random_seed,strategy=1,range_clusters=10):
	## Plotting graph of Objective function vs Number of clusters for a specific strategy
    costs = compute_all_objective_costs(range_clusters=range_clusters,strategy=strategy,random_seed=random_seed)
    k = list(range(2,range_clusters+1))
    
    # Visualize objective functions
    plt.plot(k, costs)
    plt.plot(k,costs,"bo")
    plt.xticks(list(range(1,max(k)+1)),[str(i) for i in range(1,max(k)+1)])
    plt.legend(["Elbow"])
    plt.suptitle('Strategy {} Random Seed {}'.format(str(strategy),str(random_seed)))
    plt.xlabel('Number of clusters')
    plt.ylabel('Objective Cost')
    plt.savefig(os.path.join(cur_dir,"./images/Objective_function_{}_{}".format(str(strategy),str(random_seed))))
    # plt.show();

if __name__ == "__main__":
	plot_objective_function(strategy=1,range_clusters=10,random_seed=80)
	plot_objective_function(strategy=1,range_clusters=10,random_seed=20)
	plot_objective_function(strategy=2,range_clusters=10,random_seed=80)
	plot_objective_function(strategy=2,range_clusters=10,random_seed=20)


	
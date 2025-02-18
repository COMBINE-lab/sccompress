import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/Users/zhezhenwang/Documents/patro/')
from quadt_reg3 import Point, Rect, QuadTree
from matplotlib import gridspec
import csv

# DPI = 72
#np.random.seed(60)
coln = []
def tree_from_csv(file_path,idx_x = 1, idx_y = 2, idx_cell = 3,
                  step=0.1,loop = False,method = "mean",endpt= None,allgenes = True):
    coords = []
    xs = []
    ys = []
    max_abs = 0
    count = 0
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        coln = file.readline().split(",")
        col_num = len(coln)
       # print(colnames)
        mind = [0]*col_num
        maxd = [0]*col_num
        for row in csv_reader:
            if len(row) >= 3:  # Ensure there are at least two values per row
              #try:
                  # Track min and max for each column
                #print(row)
                for i in range(idx_cell,len(row)):
                  #print(mind)
                  mind[i] = min(float(row[i]),mind[i])
                  maxd[i] = max(float(row[i]),maxd[i])
                    
                x, y= float(row[idx_x]), float(row[idx_y])
                    #max_abs = max(max_abs, *coords)
                #cells  = []
                if endpt is None:
                  if allgenes:
                    endpt = len(row)
                  else:
                    endpt = idx_cell + 1 
                #for i in range(idx_cell,endpt):
                cells = [float(value) for value in row[idx_cell:endpt]]
                coords.append([x, y,cells])
                #print(row)
                xs.append(x)
                ys.append(y)
                    # coordinates.append((x, y, z))
                count += 1
             # except ValueError:
                #print(f"Skipping invalid row: {row}")  # Handle invalid data gracefully
    #print(count)
    minx = min(*xs)-1
    miny = min(*ys)-1
    maxx = max(*xs)+1
    maxy = max(*ys)+1
    w = maxx - minx
    h = maxy - miny
    points = [Point(*coord) for coord in coords]
    domain = Rect(minx+w/2, miny+h/2, w, h)
    y_points = []
    maxerrorsl = []
    qtree = QuadTree(domain,points)
    #print(maxdata)
    if loop:
      sequence = np.arange(0, 1, step)
      for x in sequence:
        maxerror = qtree.divide(x,method,mind,maxd,maxerrors=[])
        print(x)
        print(maxerror)
        y = qtree.non0blocks()
        y_points.append(y)
        maxerrorsl.append(maxerror)
      return(maxerrorsl,y_points)
    else:
      maxerrors = qtree.divide(step,method,mind,maxd,maxerrors=[])
      return(maxerrors,qtree)
      #print(y)
      #print(qtree.points)
    # qtree.aggregate(domain)
    # fig = plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
    # ax = plt.subplot()
    # ax.set_xlim(minx,maxx)
    # ax.set_ylim(miny,maxy)
    # qtree.draw(ax)
    # # ax.scatter([p.x for p in points], [p.y for p in points], s=4)
    # #ax.invert_yaxis()
    # plt.tight_layout()
    # plt.savefig(savename)
    # plt.show()
    

#file_path = "/Users/zhezhenwang/Documents/patro/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv"
file_path = "/Users/zhezhenwang/Documents/patro/merfish6k.csv"
#file_path = "/Users/zhezhenwang/Documents/patro/test2.csv"
selected = tree_from_csv(file_path,step = 0.5)
#sys.setrecursionlimit(1500) #increase maximum recursion depth
df = selected[1].quadtree_to_df()
df.columns = coln[0][1:len(coln[0])]
df.to_csv("~/Documents/patro/quadtreedf_maxmean0.5.csv", index=False)

#with open("/Users/zhezhenwang/Documents/patro/qdtree_allg0.5.pkl","rb") as file:
#  loaded_data = pickle.load(file)

# y_points = []
# for x in range(1,1027848,513924):
#   qtree = tree_from_csv(file_path,idx_x = 5, idx_y = 6, idx_cell = 9,maxpt = x)
#   y = qtree.depth
#   y_points.append(1027848/y)

# y_mean = tree_from_csv(file_path,idx_x = 1, idx_y = 2, idx_cell = 3,
# loop = True,method = "mean")
# y_med = tree_from_csv(file_path,idx_x = 1, idx_y = 2, idx_cell = 3,
# loop = True,method = "median")
# 
# selected = tree_from_csv(file_path,idx_x = 1, idx_y = 2, idx_cell = 3,step = 0.5)

selected = tree_from_csv(file_path,loop=True)
plt.figure(figsize=(6, 6))  # Optional: Set the figure size
#step=0.1
y_mean = selected
plt.scatter(y_mean[0], y_mean[1], color="blue", marker="o", label="qdtree")
plt.show()
#plt.scatter(y_med[0], y_med[1], color="blue", marker="o", label="median")
#se_mean = [0.128,0.128,0.128, 0.129, 0.132,0.054,0.077,0.107,0.097,0.143]
#se_median = [0.1278582,0.1278582,0.1278582,0.1291324,0.1317702,0.05409406,0.1142056,0.07731104,0.1030225,0.0639858]
# based on the number of updated quadtree with all genes
plt.scatter(selected[0], selected[1], color="blue", marker="o", label="quadtree")
se_mean = [0.1278582,0.1278582,0.1278582,0.1278582,0.1278582, 0.007186814, 0.007186814, 0.1188337, 0.1503888,0.138239]
plt.scatter(se_mean, selected[1], color="red", marker="o", label="seraster")

# Add labels and title
plt.xlabel("average error rate")
plt.ylabel("quadtree # blocks")
#plt.title("Scatter Plot of Points")
plt.legend()  # Add legend
plt.grid(True)  # Add grid for better visualization
# Add numbers (data labels) to each point
# for i in range(len(se_mean)):
#     plt.annotate(f"{round(se_mean[i],3)}", (se_mean[i], selected[1][i]),
#     textcoords="offset points", xytext=(0, 5), ha='center')

# Display the plot
plt.show()

# plt.boxplot(y_points[0])
# plt.show()

# dfs = pd.DataFrame(columns=["x", "y", "value"])
# nblocksl = []
# for i in range(3,158):
#   print(i)
#   selected = tree_from_csv(file_path,idx_x = 1, idx_y = 2, idx_cell = i,step = 0.5)
#   nblocksl.append(selected[1].blocks())
#   df = pd.DataFrame(columns=["x", "y", "value"])
#   tmp = selected[1].quadtree_to_df(df)
#   tmp.columns.values[2] = selected[0]
#   n_rows = dfs.shape[0]
#   if n_rows > 0: 
#     dfs[selected[0]] = tmp.iloc[:, 2]
#   else:
#     dfs = tmp
# dfs.index = [f'block'+str(i) for i in range(0, n_rows)]
# 
# with open("df0.5.pkl","wb") as file:
#   pickle.dump(dfs,file)
  
  
# with open("/Users/zhezhenwang/Documents/patro/data/df0.9.pkl","rb") as file:
#   loaded_data = pickle.load(file)
#   
# loaded_data.to_csv('/Users/zhezhenwang/Documents/patro/data/qdtree0.9.csv', index=False) 

# selected = tree_from_csv(file_path,idx_x = 1, idx_y = 2, idx_cell = 4,step = 0.5)
# df1 = pd.DataFrame(columns=["x", "y", "value"])
# df1 = selected[1].quadtree_to_df(df1)
# n_rows = df1.shape[0]
# df1.index = [f'block{i}' for i in range(0, n_rows)]
# #### SOMEDE
# from somde import SomNode
# 
# df = pd.read_csv(dataname+'count.csv',sep=',',index_col=1)
# corinfo = pd.read_csv(dataname+'idx.csv',sep=',',index_col=0)
# corinfo["total_count"]=df.sum(0)
# X=corinfo[['x','y']].values.astype(np.float32)
# 
# som = SomNode(X,20)
# ndf,ninfo = som.mtx(df)
# nres = som.norm()
# result, SVnum =som.run()






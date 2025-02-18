#https://scipython.com/blog/quadtrees-2-implementation-in-python/
import numpy as np
import pandas as pd

class Point:
  """A point located at (x,y) in 2D space.

    Each Point object may be associated with a data object.

    """

  def __init__(self,x, y, data=None):
    self.x, self.y = x, y
    self.data = data

  def __repr__(self):
    return '{}: {}'.format(str((self.x, self.y)), repr(self.data))
  def __str__(self):
    return 'P({:.2f}, {:.2f})'.format(self.x, self.y)

  # def distance_to(self, other):
  #   try:
  #     other_x, other_y = other.x, other.y
  #   except AttributeError:
  #     other_x, other_y = other
  #   return np.hypot(self.x - other_x, self.y - other_y)

class Rect:
  """A rectangle centred at (cx, cy) with width w and height h."""

  def __init__(self, cx, cy, w, h):
    self.cx, self.cy = cx, cy
    self.w, self.h = w, h
    self.west_edge, self.east_edge = cx - w/2, cx + w/2
    self.north_edge, self.south_edge = cy - h/2, cy + h/2

  def __repr__(self):
    return str((self.west_edge, self.east_edge, self.north_edge,
              self.south_edge))

  def __str__(self):
    return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                                                   self.north_edge, self.east_edge, self.south_edge)

  def contains(self, point):
    """Is point (a Point object or (x,y) tuple) inside this Rect?"""

    try:
      point_x, point_y = point.x, point.y
    except AttributeError:
      point_x, point_y = point

    return (point_x >= self.west_edge and
            point_x < self.east_edge and
            point_y >= self.north_edge and
            point_y < self.south_edge)

  def intersects(self, other):
    """Does Rect object other interesect this Rect?"""
    return not (other.west_edge > self.east_edge or
                other.east_edge < self.west_edge or
                other.north_edge > self.south_edge or
                other.south_edge < self.north_edge)

  def draw(self, ax, c='k', lw=1, **kwargs):
    x1, y1 = self.west_edge, self.north_edge
    x2, y2 = self.east_edge, self.south_edge
    ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)


class QuadTree:
  """A class implementing a quadtree."""

  def __init__(self, boundary, points, depth=1,maxerror = 0):
    """Initialize this node of the quadtree.
        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.
        """
    self.boundary = boundary
    self.points = points
    self.depth = depth
# A flag to indicate whether this node has divided (branched) or not.
    self.divided = False
    self.maxerror = None

  def __str__(self):
    """Return a string representation of this node, suitably formatted."""
    sp = ' ' * self.depth * 2
    s = str(self.boundary) + '\n'
    s += sp + ', '.join(str(point) for point in self.points)
    if not self.divided:
      return s
      return s + '\n' + '\n'.join([
      sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
      sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])
      
  def query(self, boundary, found_points):
    """Find the points in the quadtree that lie within boundary."""

    if not self.boundary.intersects(boundary):
  # If the domain of this node does not intersect the search
  # region, we don't need to look in it for points.
      return False

# Search this node's points to see if they lie within boundary ...
    for point in self.points:
      if boundary.contains(point):
        found_points.append(point)
# ... and if this node has children, search them too.
    if self.divided:
      self.nw.query(boundary, found_points)
      self.ne.query(boundary, found_points)
      self.se.query(boundary, found_points)
      self.sw.query(boundary, found_points)
    return found_points
  
  def calculate_error(self,method,mind,maxd):
    cells = self.query(self.boundary,[])
    #print(cells)
    maxerrors = []
    if len(cells)>0:
      #print(cells[0].data)
      for j in range(0,len(cells[0].data)):
        if method == "median":
          block_mean = np.median([i.data[j] for i in cells])
        elif method == "mean":
          block_mean = np.mean([i.data[j] for i in cells])
        #print(cells[0].data)
        #maxd = max([x.data[j] for x in cells])
        #mind = min([x.data[j] for x in cells])
      # divider = maxd-mind
      # if divider==0: divider = divider+0.0001
        maxerror = np.mean([abs(x.data[j] - block_mean) for x in cells])/(maxd[j]-mind[j] + 0.01)
        maxerrors.append(float(maxerror))
      #print(maxerrors)
      maxerror = np.max(maxerrors)
    else:
      maxerror = 0
   # print("maxerror :" + str(maxerror))
    return maxerror

  def divide(self,thereshold,method,mind,maxd,maxerrors=[]):
      # maxerror = 0
      #print(len(self))
      #print(self.points)
      if len(self)>0: 
        maxerror = self.calculate_error(method,mind,maxd)
      else:
        maxerror = 0
      if maxerror > thereshold and len(self)>1:
   # There's room for our point without dividing the QuadTree.
   # """Divide (branch) this node by spawning four children nodes."""
        cx, cy = self.boundary.cx, self.boundary.cy
        w, h = self.boundary.w / 2, self.boundary.h / 2
# The boundaries of the four children nodes are "northwest",
# "northeast", "southeast" and "southwest" quadrants within the
# boundary of the current node.
        nw = Rect(cx - w/2, cy - h/2, w, h)
        points_nw = self.query(nw,[])
        #print(len(points))
        #print(nw)
        self.nw = QuadTree(nw,points_nw,self.depth + 1)
        
        ne = Rect(cx + w/2, cy - h/2, w, h)
        points_ne = self.query(ne,[])
        #print(points)
        self.ne = QuadTree(ne,points_ne,self.depth + 1)
        
        se = Rect(cx + w/2, cy + h/2, w, h)
        points_se = self.query(se,[])
        self.se = QuadTree(se,points_se,self.depth + 1)
        
        sw = Rect(cx - w/2, cy + h/2, w, h)
        points_sw = self.query(sw,[])
        self.sw = QuadTree(sw,points_sw,self.depth + 1)
    # print(self.depth)
        self.points = list(set(self.points)- set(points_nw+points_ne+points_sw+points_se))
        
        #if(len(self)>1):
        self.divided = True
        self.nw.divide(thereshold,method,mind,maxd,maxerrors)
        self.ne.divide(thereshold,method,mind,maxd,maxerrors)
        self.sw.divide(thereshold,method,mind,maxd,maxerrors)
        self.se.divide(thereshold,method,mind,maxd,maxerrors)
      else:
        #print("maxerror"+str(maxerror))
        #print(thereshold)
        self.divided = False
        self.maxerror = maxerror
        maxerrors.append(maxerror)
        #print(maxerrors)
        #print(len(maxerrors))
        # if not isinstance(maxerror, int): 
        #   maxerror = maxerror.item()
      return(np.mean(maxerrors))
# 
#   def insert(self, point,thereshold,median = False):
#     """Try to insert Point point into this QuadTree."""
# 
#     if not self.boundary.contains(point):
#   # The point does not lie inside boundary: bail.
#       return False
#   #  if len(self.points) < self.max_points:
#     maxerror = 0
#     if len(self)>0: maxerror = self.calculate_error(point,median)
#     #print("max:" + str(maxerror))
#     if maxerror <= thereshold:
#   # There's room for our point without dividing the QuadTree.
#       self.points.append(point)
#       #print(self.points)
#       return True
# 
# # No room: divide if necessary, then try the sub-quads.
#     if not self.divided:
#       self.divide()
# 
#     return (self.ne.insert(point,thereshold) or
#             self.nw.insert(point,thereshold) or
#             self.se.insert(point,thereshold) or
#             self.sw.insert(point,thereshold))


#   def query_circle(self, boundary, centre, radius, found_points):
#     """Find the points in the quadtree that lie within radius of centre.
#         boundary is a Rect object (a square) that bounds the search circle.
#         There is no need to call this method directly: use query_radius.
#         """
#     if not self.boundary.intersects(boundary):
#   # If the domain of this node does not intersect the search
#   # region, we don't need to look in it for points.
#       return False
# 
# # Search this node's points to see if they lie within boundary
# # and also lie within a circle of given radius around the centre point.
#     for point in self.points:
#       if (boundary.contains(point) and point.distance_to(centre) <= radius):
#         found_points.append(point)
# 
# # Recurse the search into this node's children.
#     if self.divided:
#       self.nw.query_circle(boundary, centre, radius, found_points)
#       self.ne.query_circle(boundary, centre, radius, found_points)
#       self.se.query_circle(boundary, centre, radius, found_points)
#       self.sw.query_circle(boundary, centre, radius, found_points)
#     return found_points

#   def query_radius(self, centre, radius, found_points):
#     """Find the points in the quadtree that lie within radius of centre."""
# # First find the square that bounds the search circle as a Rect object.
#     boundary = Rect(*centre, 2*radius, 2*radius)
#     return self.query_circle(boundary, centre, radius, found_points)

  def __len__(self):
    """Return the number of points in the quadtree."""
    npoints = len(self.points)
    if self.divided:
      npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
    return npoints
  
  def blocks(self):
    """Return # of blocks in the quadtree."""
    if not self.divided:
      npoints = 1
    else:
      npoints = self.nw.blocks()+self.ne.blocks()+self.se.blocks()+self.sw.blocks()
    return npoints
  
  def non0blocks(self):
    """Return # of non 0 blocks in the quadtree."""
    npoints = 0
    if not self.divided:
      if len(self.points)>0:
        npoints = 1
    else:
      npoints = self.nw.non0blocks()+self.ne.non0blocks()+self.se.non0blocks()+self.sw.non0blocks()
    return npoints
  
  # Recursive function to populate the matrix
  def quadtree_to_df(self, df = None,exp=True):
    if df is None:
      df = pd.DataFrame()
    # df = pd.DataFrame(columns=["x", "y"])
   # Assign mean value to the corresponding region in the matrix
    # df = pd.DataFrame()
    if len(self.points) != 0:
      values = [i.data for i in self.points]
      for pt in self.points:
        new_data = {"x":pt.x, "y":pt.y}
        #new_cols = {}
        if exp:
          for i in range(0,len(values[0])):
            elements = [j[i] for j in values]
          # Generate column names dynamically
          #new_cols["col"+str(i)] = np.nanmean(elements)
          new_data["col"+str(i)] = np.nanmean(elements) #pd.DataFrame(new_cols, index=[3])
        # Add list as new columns with dynamically generated names
       # new_data[new_col_names] = pd.DataFrame(new_cols)
        else:
          new_data["error"] = self.maxerror  
        df = pd.concat([df, pd.DataFrame([new_data])])
    if self.divided:
      # not Leaf node
      # Recursively process child nodes
      if exp:
        df = self.nw.quadtree_to_df(df= df)
        df = self.ne.quadtree_to_df(df= df)
        df = self.se.quadtree_to_df(df= df)
        df = self.sw.quadtree_to_df(df= df)
      else:
        df = self.nw.quadtree_to_df(df= df,exp=False)
        df = self.ne.quadtree_to_df(df= df,exp=False)
        df = self.se.quadtree_to_df(df= df,exp=False)
        df = self.sw.quadtree_to_df(df= df,exp=False)        
    return(df)
  ## this function has something wrong don't know why it won't update with change of points and reconstructing the tree
  def draw(self, ax,xs=[],ys=[]):
    """Draw a representation of the quadtree on Matplotlib Axes ax."""
    self.boundary.draw(ax)
    for i in self.points:
      xs.append(i.x)
      ys.append(i.y)
    if self.divided:
      self.nw.draw(ax,xs,ys)
      self.ne.draw(ax,xs,ys)
      self.se.draw(ax,xs,ys)
      self.sw.draw(ax,xs,ys)
    plt.scatter(xs, ys, color='r')
  #     
  # def aggregate(self, boundary):
  #   if not self.divided:
  #     for point in self.points:
  #       if boundary.contains(point):
  #         meanx = np.mean(point.x)
  #         meany = np.mean(point.y)
  #         meandata = np.mean(point.data)
  #   return((meanx,meany))

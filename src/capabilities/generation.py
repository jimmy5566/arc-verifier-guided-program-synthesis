import numpy as np
def n_cells_row(n,color,shape,row=0,col=0):
 if n<0 or row<0 or col<0 or row>=shape[0] or col+n>shape[1]:return None
 g=np.zeros(shape,dtype=int);g[row,col:col+n]=color;return g
def n_cells_column(n,color,shape,row=0,col=0):
 if n<0 or row<0 or col<0 or col>=shape[1] or row+n>shape[0]:return None
 g=np.zeros(shape,dtype=int);g[row:row+n,col]=color;return g
def rectangle(shape,top,left,height,width,color,filled=False):
 if min(top,left,height,width)<0 or top+height>shape[0] or left+width>shape[1]:return None
 g=np.zeros(shape,dtype=int);g[top:top+height,left:left+width]=color if filled else 0;g[top, left:left+width]=color;g[top+height-1,left:left+width]=color;g[top:top+height,left]=color;g[top:top+height,left+width-1]=color;return g
def solid_row(shape,row,color):return n_cells_row(shape[1],color,shape,row,0)
def solid_column(shape,col,color):return n_cells_column(shape[0],color,shape,0,col)
def mask_to_grid(mask,color,background=0):
 m=np.asarray(mask)
 if m.ndim!=2:return None
 out=np.full(m.shape,background,dtype=int);out[m.astype(bool)]=color;return out
def repeat_mask(mask,shape,anchors,color,background=0):
 m=np.asarray(mask).astype(bool)
 if m.ndim!=2 or not anchors:return None
 out=np.full(shape,background,dtype=int)
 for row,col in anchors:
  if row<0 or col<0 or row+m.shape[0]>shape[0] or col+m.shape[1]>shape[1]:return None
  out[row:row+m.shape[0],col:col+m.shape[1]][m]=color
 return out

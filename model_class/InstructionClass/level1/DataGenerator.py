import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import torch
import copy
import math
print(torch.__version__)
def bezier_curve(control_points, control_data,W,H,C,width):
    '''
    Usage : control_points : list of control points 
            control_data : list of control data
            W : Width of the image
            H : Height of the image
            C : Number of Channel 
            width : Width of the curve (list of width for each control point)
    example : bezier_curve([[0,0],[0,100],[100,100],[100,0]],[[255,0,0,0],[0,255,0,0],[0,0,255,0],[0,0,255,0]],100,100,4,[1,3,1,1])
    output : pytorch tensor of size (W,H,C) with bezier curve
    
    
    
    '''
    print("Drawing -------------------")
    print(control_points)
    print(control_data)
    print(width)
    print(W,H,C)
    def bezier_point(t, points):
        """
        Compute a point on a Bézier curve for a given parameter t and a list of control points.

        :param t: The parameter t along the Bézier curve, must be in the range [0, 1].
        :param points: A list of control points as tuples (x, y).
        :return: The point on the Bézier curve as a tuple (x, y).
        """
        if len(points) == 1:
            return points[0]  # The only control point is the point on the curve.

        # Compute the next set of control points by linearly interpolating between each adjacent pair.
        next_points = []
        for i in range(len(points) - 1):
            # Linear interpolation between points[i] and points[i+1] for parameter t.
            x = (1 - t) * points[i][0] + t * points[i + 1][0]
            y = (1 - t) * points[i][1] + t * points[i + 1][1]
            next_points.append((x, y))

        # Recursively compute the point on the Bézier curve with the next set of control points.
        return bezier_point(t, next_points)
    def get_surrounding_point(center,width_this,W,H):
        '''
        Usage : center : center point
                width : width of the curve
        output : list of surrounding points
        '''
        points = []
        points.append([round(center[0]),round(center[1])])
        for i in range(-width_this,width_this+1):
            for j in range(-width_this,width_this+1):
                points.append([round(center[0]+i),round(center[1]+j)])
                
        for i in points:
            if i[0] < 0:
                i[0] = 0
            if i[0] >= W:
                i[0] = W-1
            if i[1] < 0:
                i[1] = 0
            if i[1] >= H:
                i[1] = H-1
            
        return points
    def remove_duplicate_lists_efficient(input_list):
        """
        Remove duplicate 1D lists from a 2D list while preserving order in an efficient manner.
        Utilizes a dictionary to maintain insertion order and uniqueness.

        :param input_list: A 2D list containing 1D lists.
        :return: A new 2D list with duplicates removed.
        """
        seen = {}
        for sublist in input_list:
            # Use a frozenset as the key because lists are unhashable, but we need to preserve the order of elements, hence the value is the original list
            key = frozenset(sublist)
            if key not in seen:
                seen[key] = sublist
        return list(seen.values())

    beizer_mask = torch.zeros((W, H), dtype=torch.float32)  # Initialize output tensor with RGBA channels
    ## iter throgh 1000 points
    mask_point = []
    surrending_points = []
    total_no_control_points = len(control_points)
    
    for i in range(100):
        t = i / 100
        control_points_copy = copy.deepcopy(control_points)
        print(t)
        print(control_points_copy)
        
        x , y = bezier_point(t, control_points_copy)
        print(x,y)
        mask_point.append([x,y])
        ## interpolate width linearly on current point
        # TODO: 
        # inerpolate point on input width , now usage
        width_pt = int(sum(width)/total_no_control_points)
        # print(width_pt)
        if width_pt <3:
            width_pt = 1


        surrending_points.extend(get_surrounding_point([x,y],width_pt,W,H))
    
    # print(mask_point)
    # print(surrending_points)
    ## remove duplicates
    # print(len(surrending_points))
    print(surrending_points)
    for i in range(len(surrending_points)):
        beizer_mask[surrending_points[i][0],surrending_points[i][1]] = 1
    ## save the mask as png
    plt.imshow(beizer_mask)
    
    plt.savefig('bezier_mask.png')
    print(f"width_pt : {width_pt}")
    for i in range(W):
        for j in range(H):
            print(int(beizer_mask[i,j]),end="")
        print()





def render_bezier(X1, X2,debug=False):
    B, W, H, N = X1.shape
    if debug:
        print(X1.shape)
    output = torch.zeros((B, W, H, 4), dtype=torch.float32)  # Initialize output tensor with RGBA channels

    for b in range(B):
        control_points = []
        for n in range(N):
            # Find the point with the highest probability in the distribution
            probabilities = X1[b, :, :, n]
            W_max, H_max = torch.where(probabilities == torch.max(probabilities))
            if debug:
                print(W_max, H_max)
            W_max = int(W_max[0])
            H_max = int(H_max[0])
            # mix with X2 
            control_points.append([W_max, H_max])
            if debug:
                print(control_points)
        if debug:
            print("control_points")
            print(control_points)
        control_data = X2[b,:]
        if debug:
            print(control_data)
        temp = bezier_curve(control_points, control_data[:,:3], W, H, len(control_data[:,:3]), control_data[:,4])
            # Draw bezier curve using control points


    return output

# Test case and visualization setup
B, W, H, N = 1, 100, 100, 4  # Batch size, Width, Height, Number of points in Bezier curve
X1 = torch.rand((B, W, H, N)) * 0.2  # Simulated DNN output distributions
# X1[0][0][0][0] = 0.99
# X1[0][50][50][1] = 0.99
# X1[0][99][0][2] = 0.99
# X1[0][99][99][3] = 0.99
X2 = torch.rand((B, N, 5))  # Random RGBA values and widths
X2[:, :, :4] = X2[:, :, :4] * 255  # Scale the width values
output_tensor = render_bezier(X1, X2,debug=True)

# Visualization
plt.figure(figsize=(6, 6))
plt.imshow(output_tensor[0].detach().numpy())
plt.axis('off')
plt.savefig('bezier_curve.png')
plt.show()

import numpy as np
import os
from matplotlib import pyplot as plt
import statistics
import csv
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import math
from scipy import signal
#import similaritymeasures
from statistics import mean

'''
curves:
x = list[floats]
y = list[floats]
z = list[floats]
[[x,y,z...]
 [x,y,z...]
 [x,y,z...]
 ..........]

matrix:
a,b,c... = floats
[[a,b,c...]
 [d,e,f...]
 [g,h,i...]
 ..........]
'''
def direction_to_signal_2d(curve):
    '''
    x = [x1,x2,x3....]
    y = [y1,y2,y3....]
    curve = [x,y]
    '''
    curve_length = len(curve[0])
    number_of_dim = len(curve) # == 2
    sig = []
    '''
    angle = math.atan2(y2, x2) - math.atan2(y1, x1)
    '''
    for i in range(2, curve_length):
        x2 = (curve[0][i]-curve[0][i-1])
        y2 = (curve[1][i]-curve[1][i-1])
        x1 = (curve[0][i-1]-curve[0][i-2])
        y1 = (curve[1][i-1]-curve[1][i-2])
        sig.append(math.atan2(y2, x2) - math.atan2(y1, x1))
    return sig


def curve_to_signal(curve, signal_type):
    curve_length = len(curve[0])
    number_of_dims = len(curve)
    if signal_type == 'global_center':
        sig = []
        global_center_location = []
        for gcl_j in range(0, number_of_dims):
            global_center_location.append(sum(curve[gcl_j])/curve_length)
        '''
        centerx = sum(curve[0])/curve_length
        centery = sum(curve[1])/curve_length
        '''
        for i in range(0, curve_length):
            gc_signals_sum = 0
            for gc_j in range(0, number_of_dims):
                gc_signals_sum += (curve[gc_j][i] - global_center_location[gc_j])**2
            sig.append(math.sqrt(gc_signals_sum))
        return sig
    elif signal_type == 'local_center':
        sig = []
        ratio = 5
        local_length = curve_length//ratio
        startp = 0
        endp = startp + local_length
        while endp <= curve_length:
            local_center_location = []
            for lcl_j in range(0, number_of_dims):
                local_center_location.append(sum(curve[lcl_j][startp:endp])/local_length)
            '''
            centerx = sum(curve[0][startp:endp])/ratio
            centery = sum(curve[1][startp:endp])/ratio
            '''
            for i in range(startp, endp):
                lc_signals_sum = 0
                for lc_j in range(0, number_of_dims):
                    lc_signals_sum += (curve[lc_j][i] - local_center_location[lc_j])**2
                sig.append(math.sqrt(lc_signals_sum))
            startp += 1
            endp += 1
        return sig
    elif signal_type == 'velocity_center':
        sig = []
        ratio_v = 5
        local_length = curve_length//ratio_v
        startp = 0
        endp = startp + local_length
        while endp <= curve_length:
            velocity_center_location = []
            for vcl_j in range(0, number_of_dims):
                velocity_center_location.append(curve[vcl_j][startp])
            '''
            centerx = curve[0][startp]
            centery = curve[1][startp]
            '''
            for i in range(startp, endp):
                vc_signals_sum = 0
                for vc_j in range(0, number_of_dims):
                    vc_signals_sum += (curve[vc_j][i] - velocity_center_location[vc_j])**2
                sig.append(math.sqrt(vc_signals_sum))
            startp += 1
            endp += 1
        return sig
    else:
        raise ValueError('signal type error')

def align_signals(if_align, sig1, sig2):
    if not if_align:
        return (sig1, sig2)
    m1 = mean(sig1)
    m2 = mean(sig2)
    sig1 = [i - m1 for i in sig1]
    sig2 = [i - m2 for i in sig2]
    corr = list(signal.correlate(sig1, sig2))
    max_index = corr.index(max(corr))
    middle_index = len(corr)//2
    if max_index == middle_index:
        return(sig1, sig2)
    elif max_index < middle_index:
        return(sig1[0:max_index+1],sig2[len(sig2)-(max_index+1):])
    else:
        mod_index = max_index - middle_index
        return(sig1[mod_index:], sig2[0:len(sig2)-mod_index])

def calculate_distance_matrix(if_align, curves, n_curves, affinity):
    mat = np.zeros([n_curves, n_curves])
    for row in range(0, n_curves):
        for col in range(0, n_curves):
            if row == col:
                mat[row][col] = float('inf')
            else:
                if affinity in ["global_center", "velocity_center", "local_center"]:
                    original_sig_row = curve_to_signal(curves[row], affinity)
                    original_sig_col = curve_to_signal(curves[col], affinity)
                    signal_row, signal_col = align_signals(if_align, original_sig_row, original_sig_col)
                    mat[row][col] = 1-cosine_similarity([signal_row], [signal_col])[0][0]
                else:
                    '''
                    if affinity == "Frechet":
                        mat[row][col] = similaritymeasures.frechet_dist(curves[row], curves[col])
                    if affinity == "Euclidean":
                    if affinity == "Hausdorff":
                    '''
    return mat

def calculate_user_defined_matrix(curve_id, curve_startp, curve_endp, curves, n_curves):
    '''
    retmat = [(start,end) <- curve 1, (), () ...]
    '''
    curve_endp += 1
    target_sig = []
    target_center_location = []
    ret = []
    target_direction_curve = []
    for i in range(0, len(curves[curve_id])):
        target_center_location.append(sum(curves[curve_id][i][curve_startp:curve_endp]) / (curve_endp - curve_startp))
        target_direction_curve.append(curves[curve_id][i][curve_startp:curve_endp])

    for i in range(curve_startp, curve_endp):
        print("haha")
        lc_signals_sum = 0
        for j in range(0, len(curves[curve_id])):
            lc_signals_sum += (curves[curve_id][j][i] - target_center_location[j])**2
        target_sig.append(math.sqrt(lc_signals_sum))
    target_direction_sig = direction_to_signal_2d(target_direction_curve)
    print(target_sig)
    for row in range(0, n_curves):
        ret.append(user_defined_curve_to_signal(target_sig, curves[row]))
    return ret

def user_defined_curve_to_signal(input_signal, curve):
    local_length = len(input_signal)
    print("local length")
    print(local_length) 
    startp = 0
    endp = startp + local_length
    ret = None
    minv = float('inf')
    number_of_dims = len(curve)
    while endp <= len(curve[0]):
        local_center_location = []
        sig = []
        for lcl_j in range(0, number_of_dims):
            local_center_location.append(sum(curve[lcl_j][startp:endp])/(endp - startp))
        for i in range(startp, endp):
            lc_signals_sum = 0
            for lc_j in range(0, number_of_dims):
                lc_signals_sum += (curve[lc_j][i] - local_center_location[lc_j])**2
            sig.append(math.sqrt(lc_signals_sum))
        curr_distance = (1-cosine_similarity([input_signal], [sig])[0][0]) + abs(0.00000001*(input_signal[0]-sig[0]))
        if curr_distance < minv:
            minv = curr_distance
            ret = (startp, endp-1)
        startp += 1
        endp += 1
    return ret

def user_defined_curve_to_signal_add_directions(input_direction_signal, input_signal, curve):
    local_length = len(input_signal)
    startp = 0
    endp = startp + local_length
    ret = None
    minv = float('inf')
    number_of_dims = len(curve)
    while endp <= len(curve[0]):
        local_center_location = []
        sig = []
        for lcl_j in range(0, number_of_dims):
            local_center_location.append(sum(curve[lcl_j][startp:endp])/local_length)
        for i in range(startp, endp):
            lc_signals_sum = 0
            for lc_j in range(0, number_of_dims):
                lc_signals_sum += (curve[lc_j][i] - local_center_location[lc_j])**2
            sig.append(math.sqrt(lc_signals_sum))
        tempcurve = []
        for dim in range(0, number_of_dims):
            tempcurve.append(curve[dim][startp:endp])
        dsig = direction_to_signal_2d(tempcurve)
        curr_distance = 1-cosine_similarity([input_signal], [sig])[0][0]
        curr_direction_distance = (1-cosine_similarity([input_direction_signal], [dsig])[0][0])/10
        curr_distance = curr_direction_distance
        if curr_distance < minv:
            minv = curr_distance
            ret = (startp, endp)
        startp += 1
        endp += 1
    return ret


def find_minimum_distance(input, number_of_curves):
    current_min = float('inf')
    row = float('inf')
    col = float('inf')

    for i in range(0, number_of_curves):
        for j in range(0, i+1):
            if(input[i][j]<=current_min):
                current_min = input[i][j]
                row = i
                col = j
    return (row, col)

def find_clusters(input,method_name):
    all_result = []
    current_result = []
    number_of_curves = input.shape[0]
    np.fill_diagonal(input,float('inf'))

    for i in range(number_of_curves):
        current_result.append(i)
        
    all_result.append(current_result.copy())

    for _ in range(0, number_of_curves-1):
        row,col = find_minimum_distance(input, number_of_curves)
        if(method_name == "single"):
            for i in range(0,number_of_curves):
                if(i != col and i != row):
                    merged_distance = min(input[col][i],input[row][i])
                    input[col][i] = merged_distance
                    input[i][col] = merged_distance

        elif(method_name == "complete"):
            for i in range(0,number_of_curves):
                if(i != col and i != row):
                    merged_distance = max(input[col][i],input[row][i])
                    input[col][i] = merged_distance
                    input[i][col] = merged_distance

        elif(method_name == "average"):
            for i in range(0,number_of_curves):
                if(i != col and i != row):
                    merged_distance = (input[col][i]+input[row][i])/2
                    input[col][i] = merged_distance
                    input[i][col] = merged_distance
        # remove the merged curve from distance matrix
        temp_mat1 = np.delete(input, row, 0)
        temp_mat2 = np.delete(temp_mat1, row, 1)
        input = temp_mat2
        number_of_curves = input.shape[0]
       
        for n in range(len(current_result)):
            if(current_result[n] == row):
                current_result[n] = col
            elif(current_result[n] > row):
                current_result[n] -= 1
        all_result.append(current_result.copy())

    return all_result

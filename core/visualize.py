import json
import argparse
import logging
import os
import numpy as np
import sys
import matplotlib.pylab as plt


def visualization_dcgnn_tf(pts, pts_pred, seg, seg_pred_val, model_id, batch_idx, output_dir):
    color_map_file = os.path.join('./datasets/', 'part_color_mapping.json')
    color_map = json.load(open(color_map_file, 'r'))
    # print(color_map)
    # for i, m in enumerate(model_id): # batch
        # print(batch_idx, i, m, output_dir)
    output_color_point_cloud(pts[0], seg[0], color_map, os.path.join(output_dir, '%d_%s_gt.obj' % (batch_idx, model_id[0])))
    output_color_point_cloud(pts[0], seg_pred_val[0], color_map, os.path.join(output_dir,'%d_%s_pred.obj' % (batch_idx, model_id[0])))
    output_color_point_cloud(pts_pred[0], seg_pred_val[0], color_map, os.path.join(output_dir,'%d_%s_points_pred.obj' % (batch_idx, model_id[0])))
    output_color_point_cloud_red_blue(pts[0], np.int32(seg[0] == seg_pred_val[0]),
                                  os.path.join(output_dir, '%d_%s_diff.obj' % (batch_idx, model_id[0])))

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def output_color_point_cloud(data, seg, color_map, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        # print(l, data.shape, seg.shape)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def visualization(visu, visu_format, data, pred, seg, label, partseg_colors, class_choice):
    global class_indexs
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        classname = class_choices[int(label[i])]
        class_index = class_indexs[int(label[i])]
        if visu[0] != 'all':
            if len(visu) != 1:
                if visu[0] != classname or visu[1] != str(class_index):
                    skip = True
                else:
                    visual_warning = False
            elif visu[0] != classname:
                skip = True
            else:
                visual_warning = False
        elif class_choice != None:
            skip = True
        else:
            visual_warning = False
        if skip:
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1
        else:
            if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname):
                os.makedirs('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname)
            for j in range(0, data.shape[2]):
                RGB.append(partseg_colors[int(pred[i][j])])
                RGB_gt.append(partseg_colors[int(seg[i][j])])
            pred_np = []
            seg_np = []
            pred_np.append(pred[i].cpu().numpy())
            seg_np.append(seg[i].cpu().numpy())
            xyz_np = data[i].cpu().numpy()
            xyzRGB = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB_gt)), axis=1)
            IoU = calculate_shape_IoU(np.array(pred_np), np.array(seg_np), label[i].cpu().numpy(), class_choice, visual=True)
            IoU = str(round(IoU[0], 4))
            filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_pred_'+IoU+'.'+visu_format
            filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_gt.'+visu_format
            if visu_format=='txt':
                np.savetxt(filepath, xyzRGB, fmt='%s', delimiter=' ')
                np.savetxt(filepath_gt, xyzRGB_gt, fmt='%s', delimiter=' ')
                print('TXT visualization file saved in', filepath)
                print('TXT visualization file saved in', filepath_gt)
            elif visu_format=='ply':
                xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
                xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath)
                vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath_gt)
                print('PLY visualization file saved in', filepath)
                print('PLY visualization file saved in', filepath_gt)
            else:
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % (visu_format))
                exit()
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1


def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False, marker='.', s=8, alpha=0.8, figsize=(15, 7), elev=20,
                        azim=240, axis=None, title=None, *args, **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig


def plot_fig(data, save_path, model_idx):
    fig = plt.figure(figsize=(30,10))
    ax_part = fig.add_subplot(131, projection='3d')
    ax_pred = fig.add_subplot(132, projection='3d')
    ax_orig = fig.add_subplot(133, projection='3d')
    num = 0

    plot_3d_point_cloud(data['part'][num][:,0],
                        data['part'][num][:,1],
                        data['part'][num][:,2], in_u_sphere=True,axis=ax_part,show=False)
    plot_3d_point_cloud(data['pred'][num][:,0],
                        data['pred'][num][:,1],
                        data['pred'][num][:,2], in_u_sphere=True,axis=ax_pred,show=False)
    plot_3d_point_cloud(data['original'][num][:,0],
                        data['original'][num][:,1],
                        data['original'][num][:,2], in_u_sphere=True,axis=ax_orig,show=False)
    plt.savefig(os.path.join(save_path, '%d-%d.jpg' % (model_idx, num)))


def plot_3d_point_cloud_seg(x, y, z, color, show=True, show_axis=True, in_u_sphere=False, marker='.', s=8, alpha=0.8, figsize=(15, 7), elev=20,
                        azim=240, axis=None, title=None, *args, **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    # print(f'plot label; {label.shape}') # 50, 2048
    # print(f'plot color; {color.shape} x: {x.shape}, color size: {len(set(color))}') # 50, 2048

    sc = ax.scatter(x, y, z, c=color, marker=marker, cmap='hsv', s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig


def plot_fig_seg(data, save_path, model_idx, labels_pred, gt_label):
    fig = plt.figure(figsize=(30,10))
    ax_part = fig.add_subplot(131, projection='3d')
    ax_pred = fig.add_subplot(132, projection='3d')
    ax_orig = fig.add_subplot(133, projection='3d')
    num = 0
    # print(labels_pred.shape, gt_label.shape)
    color_pred = labels_pred[num].argmax(0)
    color_gt = gt_label[num]
    # print(labels_pred[num].argmax(0))
    # print(gt_label[num])

    plot_3d_point_cloud(data['part'][num][:,0],
                        data['part'][num][:,1],
                        data['part'][num][:,2], in_u_sphere=True,axis=ax_part,show=False)
    plot_3d_point_cloud_seg(data['pred'][num][:,0],
                        data['pred'][num][:,1],
                        data['pred'][num][:,2], color_pred, in_u_sphere=True,axis=ax_pred,show=False)
    plot_3d_point_cloud_seg(data['original'][num][:,0],
                        data['original'][num][:,1],
                        data['original'][num][:,2], color_gt, in_u_sphere=True,axis=ax_orig,show=False)
    plt.savefig(os.path.join(save_path, '%d-%d.jpg' % (model_idx, num)))



if __name__ == '__main__':
    print('{:<16}'.format(f'{123}'), end='')

# -*- coding: utf-8 -*-
import sys
import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import cv2
import numpy as np
import math
import tkinter
from tkinter import Tk, Toplevel,filedialog, Label, Frame, X, SUNKEN
import tkinter as tk
import os
from scipy import interpolate 
import matplotlib.pyplot as plt

def find_intersection(img, angle, center_x, center_y):
    """
    从中心点沿指定角度方向寻找图像中的白色边界点
    
    参数:
        img: 输入的灰度图像
        angle: 搜索角度(度)
        center_x: 中心点x坐标
        center_y: 中心点y坐标
        
    返回:
        找到白色边界点时返回到中心点的距离,否则返回0
    """
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 计算射线的方向向量
    angle_rad = math.radians(angle)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)  # 注意y轴向下为正
    
    # 从中心点开始沿射线方向搜索
    r = 1
    while True:
        x = int(center_x + round(r * dx))
        y = int(center_y + round(r * dy))
        
        # 确保点在图像范围内
        if x < 0 or x >= width or y < 0 or y >= height:
            break
            
        # 如果找到白色像素点(255)
        if img[y, x] > 250:
            # 计算实际距离
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            return distance
            
        r += 1
    
    return 0

def find_CenterPoint(img):
    """通过鼠标拖动设置中心点,并实时显示十字轴线"""
    # 创建图像副本用于绘制
    display_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    clicked_point = None
    is_dragging = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_point, display_img, is_dragging
        if event == cv2.EVENT_MBUTTONDOWN:
            is_dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
            # 重置显示图像
            display_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
            # 更新点击位置
            clicked_point = (x, y)
            # 绘制红色十字轴线
            cv2.line(display_img, (x, 0), (x, display_img.shape[0]), (0, 0, 255), 1)
            cv2.line(display_img, (0, y), (display_img.shape[1], y), (0, 0, 255), 1)
            # 刷新显示
            cv2.imshow('Set Center Point', display_img)
                            
        elif event == cv2.EVENT_MBUTTONUP:
            is_dragging = False
            # 检查是否点击了确定按钮
            if 10 <= x <= 100 and 10 <= y <= 40:
                cv2.destroyWindow('Set Center Point')
                    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow('Set Center Point', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Set Center Point', 800, 600)  # 设置窗口大小为800x600
    cv2.imshow('Set Center Point', display_img)
    cv2.setMouseCallback('Set Center Point', mouse_callback)
                    
    # 等待用户操作和按键
    while True:

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 13:  # ESC或回车键退出
            break
                            
    cv2.destroyWindow('Set Center Point')
                    
    # 如果用户设置了点,返回该位置,否则返回图像中心
    if clicked_point:
        return clicked_point
    else:
        return (img.shape[1]//2, img.shape[0]//2)


def draw_curve(img):
    """在图片上选择点并连接成线"""
    # 存储选中的点
    points = []
    # 存储平滑曲线的点
    smooth_segments = []
    # 当前图像的副本,用于绘制
    drawing = img.copy()
    # 转换为彩色图像以显示彩色线条
    drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)
    # 存储掩码图像
    mask = None
    # 存储拖动点信息
    dragging_point = None
    dragging_index = -1
    # 存储上一次的点集,用于撤销
    last_points = None
    # 存储最终的黑白图像
    final_image = None
    
    def draw_smooth_curve():
        nonlocal drawing, points, smooth_segments, last_points
        if len(points) < 2:
            return
            
        # 保存当前点集用于撤销
        last_points = points.copy()
            
        # 如果已经有曲线段,将上一段的终点作为这一段的起点
        if len(smooth_segments) > 0:
            last_segment = smooth_segments[-1]
            if points[0] != last_segment[-1]:  # 如果第一个点不是上一段的终点
                points.insert(0, last_segment[-1])  # 插入上一段的终点作为起点
            
        # 当点数较少时,直接使用这些点
        if len(points) <= 3:
            smooth_segments.append(points[:])
        else:
            try:
                # 使用B样条插值平滑曲线
                points_array = np.array(points)
                # 修改k参数，确保k小于点的数量
                k = min(3, len(points) - 1)
                tck, u = interpolate.splprep([points_array[:,0], points_array[:,1]], s=0, k=k)
                unew = np.linspace(0, 1, 100)
                smooth_points = interpolate.splev(unew, tck)
                
                # 将当前点集的平滑曲线保存
                current_smooth = []
                for i in range(len(smooth_points[0])):
                    current_smooth.append((int(smooth_points[0][i]), int(smooth_points[1][i])))
                smooth_segments.append(current_smooth)
            except Exception as e:
                # 如果插值失败，直接使用原始点
                print(f"平滑曲线失败，使用原始点: {str(e)}")
                smooth_segments.append(points[:])
        
        # 清空当前点集,准备接收新的点
        points = []
        
        # 重新绘制
        redraw_all()
    
    def redraw_all():
        nonlocal drawing, mask, final_image
        # 重置绘图
        drawing = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        
        # 如果存在掩码,应用掩码
        if mask is not None:
            # 创建黑白图像
            final_image = np.zeros(img.shape[:2], dtype=np.uint8)
            final_image[mask == 255] = 0  # 黑色
            final_image[mask == 0] = 255  # 白色
            
            # 在彩色显示图上显示掩码效果
            drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
            drawing_gray[mask == 255] = 0
            drawing_gray[mask == 0] = 255
            drawing = cv2.cvtColor(drawing_gray, cv2.COLOR_GRAY2BGR)
        
        # 绘制所有已保存的平滑曲线段
        for segment in smooth_segments:
            for i in range(len(segment)-1):
                cv2.line(drawing, segment[i], segment[i+1], (0, 255, 0), 1)
        
        # 绘制当前点集和折线
        for i, pt in enumerate(points):
            cv2.circle(drawing, pt, 3, (255, 0, 0), -1)
            if i > 0:
                cv2.line(drawing, points[i-1], pt, (0, 0, 255), 1)
    
    def find_nearest_point(x, y, threshold=5):
        # 检查当前点集
        for i, (px, py) in enumerate(points):
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < threshold:
                return (px, py), i
        return None, -1
    
    def create_mask():
        nonlocal mask, drawing
        if len(smooth_segments) < 1:
            return
            
        # 创建空白掩码
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # 收集所有点形成闭合轮廓
        contour = []
        for segment in smooth_segments:
            contour.extend(segment)
        
        # 确保轮廓闭合
        if contour[0] != contour[-1]:
            contour.append(contour[0])
            
        # 将点列表转换为numpy数组格式
        contour = np.array(contour, dtype=np.int32)
        
        # 填充轮廓
        cv2.fillPoly(mask, [contour], 255)
        
        # 重新绘制
        redraw_all()
        
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, dragging_point, dragging_index
        
        if event == cv2.EVENT_MBUTTONDOWN:
            # 检查是否点击到已有点
            nearest, idx = find_nearest_point(x, y)
            if nearest:
                dragging_point = nearest
                dragging_index = idx
            else:
                # 如果已经有曲线段且这是新段的第一个点
                if len(smooth_segments) > 0 and len(points) == 0:
                    last_segment = smooth_segments[-1]
                    points.append(last_segment[-1])  # 添加上一段的终点作为起点
                
                # 添加新点
                points.append((x, y))
                redraw_all()
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_point is not None:
                # 更新拖动点的位置
                points[dragging_index] = (x, y)
                redraw_all()
                
        elif event == cv2.EVENT_MBUTTONUP:
            dragging_point = None
            dragging_index = -1
            
        cv2.imshow('Draw Curve', drawing)
    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow('Draw Curve')
    cv2.setMouseCallback('Draw Curve', mouse_callback)
    
    # 等待用户操作
    while True:
        cv2.imshow('Draw Curve', drawing)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 13:  # ESC键或Enter键退出
            break
        elif key == ord('p'):  # p键平滑当前曲线段
            draw_smooth_curve()
        elif key == ord('m'):  # m键填充闭合区域
            create_mask()
        elif key == ord('z'):  # z键撤销平滑
            if last_points:
                # 删除最后一段平滑曲线
                if len(smooth_segments) > 0:
                    smooth_segments.pop()
                points = last_points
                last_points = None
                redraw_all()
    
    cv2.destroyAllWindows()
    return final_image

def process_image(image_path):
    try:
        # 读取图像
        # 使用 cv2.imdecode 和 np.fromfile 来读取图像文件
        # 这种方式可以支持中文路径,并将图像转换为灰度格式
        img_array = np.fromfile(image_path, np.uint8)  # 将图像文件读取为字节数组
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # 解码为灰度图像
        
        if img is None:
            raise Exception("无法读取图像，请确认文件格式是否正确")

        # 通过鼠标拖动设置中心点
        center_x, center_y = find_CenterPoint(img)
        print(f"中心点: {center_x}, {center_y}")
        processed_img = draw_curve(img)
        
        # 确保0度方向的距离为参考值
        reference_distance = find_intersection(processed_img, 0, center_x, center_y)
        if reference_distance == 0:
            raise Exception("无法到参考点，请确认图像是正确")
        
        distances = []
        for angle in range(360):
            distance = find_intersection(processed_img, angle, center_x, center_y)
            # 归一化距离，确保0度方向为1
            normalized_distance = distance / reference_distance if distance > 0 else 0
            distances.append(normalized_distance)
            
            if angle % 45 == 0:
                print(f"处理进度: {angle}度, 距离值: {normalized_distance:.4f}")
        
        return distances
        
    except Exception as e:
        raise Exception(f"处理图像时出错: {str(e)}")

def main():
    try:
        # 创建启动画面窗口
        splash = Tk()
        splash.title("天线图转天线增益数据表程序")
        splash.geometry("400x200")
        
        # 添加标题文本
        title_label = tkinter.Label(splash, text="天线图转天线增益数据表程序", font=("Arial", 16, "bold"))
        title_label.pack(pady=40)
        
        # 添加分隔线
        separator = Frame(splash, height=2, bd=1, relief=SUNKEN)
        separator.pack(fill=X, padx=20, pady=20)
        
        # 显示3秒后自动关闭
        splash.after(2000, splash.destroy)
        splash.mainloop()
        
        # 创建Tk根窗口
        root = Tk()
        # 隐藏主窗口
        root.withdraw()
        
        # 打开文件选择对话框
        image_path = filedialog.askopenfilename(
            title='选择天线方向图图片',
            filetypes=[
                ('图片文件', '*.png;*.jpg;*.jpeg;*.bmp'),
                ('所有文件', '*.*')
            ]
        )
        
        # 如果用户没有选择文件就退出
        if not image_path:
            print("未选择文件")
            return
            
        # 显示选择的文件路径，帮助调试
        print(f"选择的文件: {image_path}")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print("错误：选择的文件不存在")
            return
            
        distances = process_image(image_path)
        
        # 让用户选择保存结果的位置
        save_path = filedialog.asksaveasfilename(
            title='保存数据文件',
            defaultextension='.txt',
            filetypes=[('文本文件', '*.txt'), ('所有文件', '*.*')]
        )
        
        # 如果用户没有选择保存位置就退出
        if not save_path:
            print("未选择保存位置")
            return
            
        # 将结果保存到文件
        with open(save_path, "w", encoding='utf-8') as f:
            for angle, distance in enumerate(distances):
                f.write(f"{angle},{distance:.4f}\n")
        
        print(f"处理完成！数据已保存到 {save_path}")

        # 对0值进行值处理
        with open(save_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        data = []
        for line in lines:
            angle, value = line.strip().split(',')
            data.append([int(angle), float(value)])
            
        # 遍历数据进行插值
        for i in range(len(data)):
            if data[i][1] == 0:  # 如果找到0值
                # 向前找最近的非0值
                prev_val = 0
                prev_idx = i - 1
                while prev_idx >= 0:
                    if data[prev_idx][1] != 0:
                        prev_val = data[prev_idx][1]
                        break
                    prev_idx -= 1
                
                # 向后找最近的非0值
                next_val = 0  
                next_idx = i + 1
                while next_idx < len(data):
                    if data[next_idx][1] != 0:
                        next_val = data[next_idx][1]
                        break
                    next_idx += 1
                    
                # 如果找到了前后的非0值,进行线性插值
                if prev_val != 0 and next_val != 0:
                    data[i][1] = (prev_val + next_val) / 2
                # 如果只找到前值,使用前值
                elif prev_val != 0:
                    data[i][1] = prev_val
                # 如果只找到后值,使用后值    
                elif next_val != 0:
                    data[i][1] = next_val
                    
        # 将处理后的数据写回文件
        with open(save_path, 'w', encoding='utf-8') as f:
            for angle, value in data:
                f.write(f"{angle},{value:.4f}\n")

        
        # 从保存的文件中读取数据并绘制图形
        import matplotlib.pyplot as plt
        
        # 读取数据
        angles = []
        values = []
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                angle, value = line.strip().split(',')
                angles.append(float(angle))
                values.append(float(value))
                
        # 创建极坐标图
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # 将角度转换为弧度
        angles_rad = np.deg2rad(angles)
        
        # 绘制数据
        ax.plot(angles_rad, values)
        
        # 设置图表属性
        ax.set_title('天线方向图', pad=20)
        ax.grid(True)
        
        # 显示图形
        plt.show()
        
    except Exception as e:
        print(f"错误：{str(e)}")
        
    finally:
        # 等待用户输入后再关闭
        input("按回车键退出...")

if __name__ == "__main__":
        # 创建根窗口但不显示
    root = tk.Tk()
    root.withdraw()
    
    # 创建启动画面
    splash = Toplevel(root)
    splash.title("")
    splash.overrideredirect(True)  # 移除标题栏
    
    # 获取屏幕尺寸
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    
    # 设置启动画面尺寸和位置
    width = 500
    height = 100
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    splash.geometry(f"{width}x{height}+{x}+{y}")
    
    # 设置整个窗口背景为黑色
    splash.configure(bg='black')
    
    # 创建一个Frame来容纳标题和作者
    title_frame = tk.Frame(splash, bg='black')
    title_frame.pack(expand=True, fill="both")
    
    # 创建标题容器
    title_container = tk.Frame(title_frame, bg='black')
    title_container.pack(expand=True, fill="both")
    
    # 添加标题文本,居中显示
    tk.Label(title_container, text="天线图转天线增益数据表工具", font=("黑体", 20, "bold"), 
            bg='black', fg='white').pack(expand=True)
    
    # 创建作者容器
    author_container = tk.Frame(title_frame, bg='black')
    author_container.pack(fill="x", padx=5, pady=5)
    
    # 添加作者名字,右下角显示
    tk.Label(author_container, text="HXDI 2024/11/22", font=("黑体", 10, "italic"),
            bg='black', fg='white').pack(expand=True)
    
    # 更新显示
    splash.update()
    
    # 延迟1秒后关闭启动画面并启动主程序
    def start_main():
        splash.destroy()
        root.destroy()
        main()
        sys.exit()  # 确保程序完全退出
        
    splash.after(1000, start_main)
    
    # 启动主循环
    splash.mainloop()
     
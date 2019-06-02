# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:38:43 2019

@author: 10670
"""
import sys; sys.path
from tkinter import*
import GUI_CeShi

def main():
    root = Tk()
    root.title("手写数字识别系统")
    
    #生成标签
    main_label = Label(root, text = "欢迎来到手写数字识别系统")
    main_label.grid(row = 0, column = 0, padx = 30, pady = 5)

    #生成提示信息标签
    choose_label = Label(root, text = "点击开始按钮进入系统")
    choose_label.grid(row = 1, column = 0, padx = 30, pady = 5)
    
    #生成开始按钮
    add_button = Button(root, text = "开始", bg = "light blue", command = GUI_CeShi.main)
    add_button.grid(row = 2, column = 0, padx = 30, pady = 5)

    root.mainloop()
def shibie():
    pass
if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:53:46 2019

@author: 10670
"""

from tkinter import*
import application

def main():
    sort_root = Tk()
    #此时在主函数调用时，sort窗口可以独立弹出，而不是附着在主窗口中
    sort = App(master = sort_root) 
    
    sort.master.title("识别")
    sort_root.mainloop()
    
class App(Frame):
    def __init__(self, master = None):
        
        Frame.__init__(self, master)
        self.grid()
        
        self.PicPath = StringVar()
        
        self.creat_label()  #创建标签
        self.creat_entry()  #创建输入框
        self.creat_button() #创建按钮
        
        self.confirm()
    
    def creat_label(self):
        Label(self, text = "请输入要识别的图片链接"). \
        grid(row = 0, column = 0, padx = 5, pady = 5, sticky = E)

    def creat_entry(self):
        Entry(self, width = 20, textvariable = self.PicPath).grid(row = 1, column = 0,padx = 5)
        
    def creat_button(self):
        Button(self, text = "开始识别", command = self.confirm). \
        grid(row = 2, columnspan = 2, padx = 5, pady = 5)
    def confirm(self):
        a = self.PicPath.get()
        print(a)
        print(type(a))
        #application.pre_pic(a)
        b = 'D:\pic\0.png'
#        application.pre_pic(b)
        application.pre_pic(r'D:\pic\0.png')
        #application.pre_pic(self.PicPath.get())
if __name__ == "__main__":
    main()
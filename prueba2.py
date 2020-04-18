import tkinter as tk
from tkinter import *

def ingresar():
    ventanaIngresar = tk.Toplevel(ventanaPrincipal)
    ventanaIngresar.title("Ingresar")
    ventanaIngresar.geometry("350x300")
    tk.Label(ventanaIngresar, text="Tipo de inversion",
             font=("Agency FB", 14)).place(x=50, y=50)
    btn= Button(ventanaIngresar, text='Nuevo usuario', command=newUser)
    btn.pack(anchor=CENTER, expand=True)

def newUser():
    print("hola")

ventanaPrincipal = tk.Tk()
ventanaPrincipal.geometry('380x200')
tk.Button(ventanaPrincipal, text="Ingresar", command=ingresar,
          font=("Agency FB", 14), width=10).place(x=130, y=30)
ventanaPrincipal.mainloop()   
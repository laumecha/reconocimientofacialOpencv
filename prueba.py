
from tkinter import *
from tkinter import messagebox as mb
from tkinter import colorchooser
from tkinter import filedialog

def uno():
	'''
	Regresa True o False -> 1 y 0
	'''
	r = mb.askokcancel("Pregunta", "Quieres un gato")
	Label(root,text=r).pack()

def dos():
	'''
	Regresa 'yes' o 'no'
	'''
	r = mb.askquestion("Pregunta", "Quieres un gato")
	Label(root,text=r).pack()

def tres():
	'''
	Regresa ok
	'''
	r = mb.showwarning("Peligro","Si no tienes 3 puntos no tendras constancia")
	Label(root,text=r).pack()	

def cuatro():
	'''
	Regresa True o False -> 1 y 0
	'''
	r = mb.askretrycancel("Error","No se puede ejecutar la accion")
	Label(root,text=r).pack()	
	
def color():
	color = colorchooser.askcolor(title="Paleta de colores")
	'''
	Regresa una tupla con los colores rgb y en hexadecimal
	'''
	Label(root,text=color[1]).pack()
	root.config(bg=color[1])	

def abrir():
	'''
	Regresa la ruta absoluta del archivo
	Si solo se agrega un tipo de archivo se debe dejar una coma 
	filetypes = (("Fichero texto","*.txt"),)
	'''
	ruta = filedialog.askopenfilename(title="Abrir",filetypes = (("Fichero texto","*.txt"),("Fichero PDF","*.pdf"),("Todos los ficheros","*.*")))
	Label(root,text=ruta).pack()
	
def guardar():
	'''
	Regresa la ruta absoluta del archivo modo de apertura = w y la codificacion (juego de caracteres)
	'''
	fichero = filedialog.asksaveasfile(title="Guardar",mode="w+",defaultextension=".py",filetypes = (("Fichero texto","*.txt"),("Fichero PDF","*.pdf"),("Todos los ficheros","*.*")))
	if fichero is not None:
		fichero.write("print('Hola Mundo - Creado desde el GUI')\n")
		fichero.write("input('ctrl+z para salir')")
		fichero.close()
	Label(root,text=fichero).pack()	
	
root = Tk()
root.geometry("300x300+500+300") # +500+300 es para indicar en que parte de la pantalla se ubicara
root.title("Crear Pop-Up")
root.config(bd=10)


Button(root,text="Click Me", command=uno).pack()
Button(root,text="Don't Click Me",command=dos).pack()
Button(root, text="No",command=tres).pack()
Button(root, text="Reintentar",command=cuatro).pack()

Button(root,text="Colores", command=color).pack()
Button(root,text="Abrir", command=abrir).pack()
Button(root,text="Guardar", command=guardar).pack()

root.mainloop()
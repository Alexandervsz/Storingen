# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:38:36 2021

@author: Gebruiker
"""

from tkinter import *
from dataProcessing import prediction, loadJobLib

root = Tk()

root.geometry("500x300")

def dataGet():
    print("Hello")


def beginClick():
    clicked = Label(root, text="Klicked.")
    clicked.grid(row=4,column=1)
    tree, kMeans = loadJobLib()

    sap_tijd = str(invoer1_1.get()) + str(invoer1_2.get()) + "00"
    monteur_tijd = str(invoer2_1.get()) + str(invoer2_2.get()) + "00"
    magie = prediction([int(sap_tijd),int(monteur_tijd)],tree)
    uitvoer.delete(0,END)
    uitvoer.insert(0,magie[0])



# Label widgets.
uurLabel = Label(root, text="uur")
minuutLabel = Label(root, text="minuut")
inText1 = Label(root, text="Meldtijd")
inText2 = Label(root, text="Aannemer Ter Plaatse")
#inText3 = Label(root, text="Text3")

# Button widgets.
beginButton = Button(root, text="Begin", command=beginClick)

# Input field widgets.
invoer1_1 = Entry(root)
invoer2_1 = Entry(root)
# invoer3_1 = Entry(root)
invoer1_2 = Entry(root)
invoer2_2 = Entry(root)
# invoer3_2 = Entry(root)

# Output widget
uitvoer = Entry(root)

# Grid positions
uurLabel.grid(row=0, column=1)
minuutLabel.grid(row=0, column=2)
inText1.grid(row=1, column=0)
inText2.grid(row=2, column=0)
# inText3.grid(row=3, column=0)

invoer1_1.grid(row=1, column=1)
invoer2_1.grid(row=2, column=1)
# invoer3_1.grid(row=3, column=1)
invoer1_2.grid(row=1, column=2)
invoer2_2.grid(row=2, column=2)
# invoer3_2.grid(row=3, column=2)
uitvoer.grid(row=5,column=2)

beginButton.grid(row=5, column =1)

root.mainloop()
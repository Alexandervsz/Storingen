# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:38:36 2021

@author: Gebruiker
"""

from tkinter import *
from dataProcessing import prediction, loadJobLib
import math
import numpy as np
import pandas as pf

root = Tk()


# root.geometry("300x400")

def setOutput(resultaat, prob, tijd, screenName):
    hours = int(resultaat / 60)
    minuten = resultaat - hours * 60
    eindtijd = str(hours + int(tijd[0:2])) + ":" + str(minuten + int(tijd[2:4]))

    outScreen = Tk()
    outScreen.title(screenName)
    VoorspellingtxtBx = Label(outScreen, text="Voorspelling")
    probtxtBx = Label(outScreen, text=str(round(prob, 2)) + "%", background="white", padx=10, pady=10)
    outScreentxtBx2 = Label(outScreen, text=resultaat, background="white", padx=10, pady=10)
    zekerheidtxt = Label(outScreen, text="Zekerheid voorspelling:", padx=10, pady=10)
    voorspellingMinuten = Label(outScreen, text="Voorspelling in minuten:")
    eindtijdLabel = Label(outScreen, text="Eindtijd")
    eindtijdtxt = Label(outScreen, text=eindtijd, background="white")

    voorspellingMinuten.grid(row=0, column=2, padx=10, pady=10)
    zekerheidtxt.grid(row=0, column=3, padx=10, pady=10)
    VoorspellingtxtBx.grid(row=1, column=1, padx=10, pady=10)
    outScreentxtBx2.grid(row=1, column=2, padx=10, pady=10)
    probtxtBx.grid(row=1, column=3)
    eindtijdLabel.grid(row=2, column=1)
    eindtijdtxt.grid(row=2, column=2)


def fixInput(input):
    if len(input) < 1:
        return "00"
    elif len(input) < 2:
        return "0" + input
    else:
        return input


def beginClick():
    tree, kMeans, poly, lr = loadJobLib()

    sap_tijd = fixInput(str(invoer1_1.get())) + fixInput(str(invoer1_2.get())) + "00"
    monteur_tijd = fixInput(str(invoer2_1.get())) + fixInput(str(invoer2_2.get())) + "00"
    monteur_prognose_tijd = (str(invoer3_2.get()))
    magie, probability = prediction([int(sap_tijd), int(monteur_tijd)], kMeans)
    prio = int(invoer4_1.get())
    if prio == 1:
        lst = [1, 0, 0, 0, sap_tijd, monteur_tijd, monteur_prognose_tijd]
    elif prio == 2:
        lst = [0, 1, 0, 0, sap_tijd, monteur_tijd, monteur_prognose_tijd]
    elif prio == 4:
        lst = [0, 0, 1, 0, sap_tijd, monteur_tijd, monteur_prognose_tijd]
    elif prio == 5:
        lst = [0, 0, 0, 1, sap_tijd, monteur_tijd, monteur_prognose_tijd]
    else:
        lst = [0, 0, 0, 0, sap_tijd, monteur_tijd, monteur_prognose_tijd]
    polyGuess = lr.predict(poly.fit_transform([lst]))[0]
    setOutput(magie[0], probability, sap_tijd, "kNN voorspelling")
    setOutput(int(polyGuess), probability, sap_tijd, "Polynomial voorspelling")


# Label widgets.
uurLabel = Label(root, text="uur")
minuutLabel = Label(root, text="minuut")
inText1 = Label(root, text="Meldtijd")
inText2 = Label(root, text="Aannemer Ter Plaatse")
inText3 = Label(root, text="Voorspelling Monteur")
inText4 = Label(root, text="Prioriteitscode:")

# Button widgets.
beginButton = Button(root, text="Begin", command=beginClick)

# Input field widgets.
invoer1_1 = Entry(root)
invoer2_1 = Entry(root)
invoer3_1 = Entry(root)
invoer1_2 = Entry(root)
invoer2_2 = Entry(root)
invoer3_2 = Entry(root)
invoer4_1 = Entry(root)

# Output widget
# uitvoer = Entry(root)

# Grid positions
uurLabel.grid(row=0, column=1)
minuutLabel.grid(row=0, column=2)
inText1.grid(row=1, column=0)
inText2.grid(row=2, column=0)
inText3.grid(row=3, column=0)
inText4.grid(row=4, column=0)

invoer1_1.grid(row=1, column=1)
invoer2_1.grid(row=2, column=1)
# invoer3_1.grid(row=3, column=1)
invoer1_2.grid(row=1, column=2)
invoer2_2.grid(row=2, column=2)
invoer3_2.grid(row=3, column=2)
invoer4_1.grid(row=4, column=1)
# uitvoer.grid(row=5,column=2)

beginButton.grid(row=5, column=1)

root.mainloop()

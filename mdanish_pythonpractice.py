#!/usr/bin/env python3

HAL = 9000

Bridge = "4"

Nahn = [2]

R2 = ["D", "2"]

C = [{"3PO"}]


Poppins = set("supercalifragilisticexpialidocious")

WALL = [["E"]]

K = {"2SO"}

Galactica = {
    "Karl": "Helo",
    "Kara": "Starbuck",
    "Lee": "Apollo",
}

Galactica["Sharon"] = "Boomer"
Galactica["Marge"] = "Racetrack"

Wheel = {
    "Blue": ["Moiraine", "Siuan"],
    "Green": ["Alanna"],
    "Purple": [],
    "White": ["Sarene"],
    "Yellow": [],
    "Plaid": [],
    "Red": ["Elaida", "Liandrin"],
}

Wheel["Brown"] = ["Verin", "Saerin", "Janya"]
Wheel["Green"].append("Egweyne")
Wheel["Red"].remove("Liandrin")
Wheel["Yellow"].append("Nynaeve")
del Wheel["Plaid"]



Germanna_levels = ["Freshman", "Sophomore"]

UMW_levels = ["Freshman", "Sophomore", "Junior", "Senior"]


MaryWash_levels = UMW_levels


CNU_levels = list(UMW_levels)


def plus2(x):
    return x + 2


_DAT_SET = {"Navani", "Shallan", "Jasnah"}

def gimme_dat_set():
    return _DAT_SET



def gimme_set_like_dat():
    return {"Kaladin", "Dalinar", "Adolin"}


def center(s):
    if s == "":
        return None
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    else:
        return s[mid - 1 : mid + 1]


def nuke_last(lst, item):

    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == item:
            del lst[i]
            return



def middlest(a, b, c):
    return sorted([a, b, c])[1]


def tack_on_end(lst, thing, times=1):
    for _ in range(times):
        if isinstance(thing, list):
            lst.extend(thing)
        else:
            lst.append(thing)



def wondrous_count(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps


def unique_vals(d):
    return list(set(d.values()))


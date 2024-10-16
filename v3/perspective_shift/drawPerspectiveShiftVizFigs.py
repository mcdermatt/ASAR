#setup - rememeber to switch to tensorflow 2.3 kernel...
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime
import trimesh
import time
from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget

#need to have these two lines to work on my ancient 1060 3gb
#  https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


plt = Plotter(N = 3, axes = 4, bg = (1, 1, 1), interactive = True)
disp1 = [] #before estimated transformation (drawn on left)
disp2 = [] #after 1 transformation (drawn in center)
disp3 = [] #after niter transformations

alph = 0.8
rad = 3

#read in dense point cloud of car
points_per_sample = 350 #num pts per scan - defined in MatLab script

#actually a human
# c1 = np.loadtxt('training_data/car_demo2_scan1.txt') 
# c2 = np.loadtxt('training_data/car_demo2_scan2.txt')
# gt = np.loadtxt('training_data/car_demo2_ground_truth.txt')
# #only keep above the shoulders
# # c1 = c1[c1[:,2] > -3.55]
# # c2 = c2[c2[:,2] > -3.55]
# c1 = c1[c1[:,2] > -3.3]
# c2 = c2[c2[:,2] > -3.3]

#human + wall
c1 = np.loadtxt('figures/fig1_s1.txt') 
c2 = np.loadtxt('figures/fig1_s2.txt')
gt = np.loadtxt('figures/fig1_gt.txt')
c1 = c1[c1[:,2] > -2] #remove ground plane
c2 = c2[c2[:,2] > -2]


mean1 = np.mean(c1, axis = 0)
mean2 = np.mean(c2, axis = 0)

#raw points
disp1.append(Points(c1 - np.array([0., -0.2, 0.3]), c = 'red', r = rad, alpha = alph))
disp1.append(Points(c2 + mean1 - mean2 + np.array([0,-0.2,0.3]), c = 'blue', r = rad, alpha = alph))

#match cloud means
disp2.append(Points(c1, c = 'red', r = rad, alpha = alph))
disp2.append(Points(c2 + mean1 - mean2 , c = 'blue', r = rad, alpha = alph))

#draw true soln
disp3.append(Points(c1, c = 'red', r = rad, alpha = alph))
disp3.append(Points(c2 - gt/10, c = 'blue', r = rad, alpha = alph))

plt.show(disp1, "Initial clouds", at = 0)
plt.show(disp2, "Matching Point Cloud Means", at = 1)
plt.show(disp3, "Correct Translation", at = 2)








# WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWNNNNNNWNNNNNNNWWWWNNNNXXXXKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK0000000KKKKK00000KKKKXXXNNNNNNNXXXXXXKKKKKKKKXXXXXKKKKXNNXXKKKKKK0000000000000000000KKK000OOOO000000KKX
# NNNNNNWWWWNNNNNNNXXXNXXXXXXXXNNXXXXXXXXKKXXXXKKKKKKKKKXKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKXXNNNNNNNNNNNXXXXXKKKKKKXXXXXXKKKKKXXXXKKKKKK000000000000000000KKK00000OOO00000000KK
# XXXKKKXXXXKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKXXKKKKKKKKKKKKKKKKKKKKKKK00KKKKKKKKKKKXXNNNNNNNNNNNXXXXKKKKKKKXXNNXXXKKKKKXKKKK0000000000000000000000KKK0000000000000000000
# KK000K000000KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK00KKKKKKKKKKKKKKKKKKXKKKKKKKKKKKKKKKKKKKKKKKKKKKKK0KKKKKKKKKKKKKKKKKKKKXXNNNNNNNNNNNXXXKKKKKKKKXXXXXXKKKKKKKKKKK00000000000000000000KKKKK00000000000000000000
# 0KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK000KKKKKKKXXXXKKKK00KKKKKKKKKKK000KKKKKKKKKKK0000KKKKKKKKKKKKKKKKXXXXXKKXXXXNNNNNNNNNNXXKKKKKKKKKKXXXXXKKK00KKKKKKK000000000000K000000KKKKKKK000000000000000KKKKK
# KKKKKKKKKKKKKKKKKKKKKKKKKKKK000KKKKKKKKKKKKKKKKKKKKKKKKKKKK0KKKKKKKKKK0000KKKKKKKKK0000KKKKKKKKKKKKXXXXXXXXXXXXXXXNNNNNNNNNXXXKKKKKKKKKXXXXXXKK000KKKKKKK000000000K000K00000KKKKKKK000000000KKKKKKKKKKKK
# 00KKKK00KKKKKKKKKKKKKK000KKKK00KK0KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK000KK000KKKKKKKKKKKKKKKKKKKKXXXXXXXXXXXXXXXXXXXXXXNNNNNNNXXKKKKKKKKXXXXXXXXKKKKKKKKKKKKKKKKKKKKKKKKKKKK0KKKKKKKKK0000000KKKKKKKKKKKKKKK
# O000000000000K00000000O00000000000000000KKKKKKKKKKK0KKKKKKKK0000K00000000KKK0000KKKKKKKKKKKKKKKKKKXXXXXXXXXKKXXXXXXXXXXXXXXKKKKKKKKKKXXKXXXXKKKKKKKKKKKKKKKKK00KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
# kO0OOO000OOOOO00000OOOOOO00OOO0000000000KK00000000000KKKKKK00KKKK000000000K000000KKK00KKKK00000KKKKKKKKKKKKKKKKKKKKKKKK000000KKKKK000KK00KK0000000000000000000000000000000000KKK0000KKKKKKK0KKKKKKKKKKKK
# OO0OOO00OOOOOOO00000OOOO0KK0OOOO0000000000000000000000KKKKK0KKXXXK000KKKKKKKKK0KKKK0000K0000000000000000000KKK00000K0000000000000000000000OOO0000OOOOOOOOOOOOOOOOOOOOkkOOOOOO00OOkOO0000000000KKKKKKKKKK
# 00KK0KK00K000KKKKKXXKK00KKK0000000000KK0000000000000000000000KKXXKKK0KKXXKK0000KKKK000KK00OOO0O00000000000KKK000000000000000OOO0000O000OOOOOOOOOOOOkkOOOOOOOOkkOOOOOkkkkkkkkkkOOOOkOOOOOkkkkOOOO000OO000
# KKKKXXXXXXXKKKXNNXXXNNXXXXXXXXXXXXXXXXXKKXXXXXXXXKKKK00KKK000KKKKXXKKKXXXKK0000KKKK0000K00000000OOOOOO000000000O000OOOOOOOOOOOOO0000000OOOOOOOOOOOkkkOOOkOOOOOOOOO00OOOkkkkkkkOOOOOkkkkxxxxxxkkkkOOOOOOO
# 0OO0KXNNNNNXKXXNNXXXNNXXXNNNXNNNNNXXNNNNNWWWWNWWWNNNNNNNNXXXXXXXKKXXXXNNNNXXXXXXXKKKKKKXXXXKKKKKKKKKKK000000000OO00OOOOOOOOOOOO000KK0000OO000000000000000KKKKKKKKKK0KK000OOOOOOOO0OOOOkkxxxxxxxxxkOOkkkk
# 0OO0KXNNNNNXXXXXKKKKXXXXXNNNNXXXXXXXNNNNWWWWWWWWWWWWWWWWWWWWWWWNNNNNNNWWWWWNNNNNXXNNNNNNNNNNNNNNNNNNNNNXXXKKKXKKKKXXKKKKKK00000KXXXXKKXKKKXXXKKKXXKKXXNNNNNNNNNXXXXXXXKKKK0kkO0000000000K0OkkkkxxkO0Okxx
# KK000KXXXXXK0000OOOO0KXXXXXXXKK0KKKKXXNNNNNNXNNNNNNNNNWWWWNNNNNWWWWWNNNNWWNXKXNNKKXNNNNNWWWWNNNWWWWWWWWWWNNXXNNNNNNNNNXXXXK0000KXXXXXXXXXNNNXKKXXK0000KXNNXNNNNNNNNNNNNNNXK00KXXXXXXXXXXXX000KK00KKK0Oxx
# XK0OOO000KKOkkkOOOkkO00KKK00000000000KKKKKKK00KKXXNNNNNNNNXXKXNNNXXXXXXXNNXK0KXXKKKXXKXNNWWNNXNNWNNNNNNNNNNNNNNNNNNNNNXXXXKK0000K00KKK00KKKXK00KKK0OOO00KKKXXXXXXXXXXXKKKXKKKXNNNNNNNNXXKOOOO0KXXKK0Okxx
# K0kkkkkkO00OkkOOOOOOO000OOOOOO000000000OO000O000KXXXXNNNNXXKKKKKK0000000KXXK0KK000KKKKKXXXXXXXXXXXKKKKXXXKKKK00OOkkkO00000KKK0OOOOOkOO0OOOOOOOOO000OOOOOOOO000000O00000000000KKKKKKKXXK0kxxxkk00K0Okkkkk
# 0OOOOOkkOOOOOO000OO0KKKK000000KKKKKK0000000000000K00O0KKKXXXKKK000OOO0000KKKKKKKKKKKKKKKKXKKKKKKXXXKXKKOkkkxxxxdxxdloxO0000KKK00OOkkkkkkkkOO00000KK00000000000000OkkO00KKKKKKKKKKKKKKXKK00000KKKKK0KKKKK
# KK0KK00000000KKKKKKKKXXXKKKKKKXXXXXXXKKKKKK00KKKKK0OO0KKKKKXXXXKK000KKKKXXXXXXXXXXXXXXKKXXXXXK0OOkxddxdolllloooodxxdodk0K0000OOkdocc::::cloxkO0KKKKKKK000KKXXKKK0xlcldOO00KKXXXXXKKKKXXXXXXKKKKKKKKKKKKK
# 000KXXXXXXXXXXXXXXXXKXXXXXXXXXXXXXXXXXXXXXXXXXXXXXKKKXXXXXXXXNNNNXXXXXNNNNXXXXXXNNXXXXXXXXXKOdcc:;,,,;clc;;;:ccc::clodkOxoclooc:::;,'',,;:cccloodxk0KK000KXNNNNX0o;,,:ldddxOKXXXXXKKXXXXXXXXXXKKKKKKKKK0
# :clx0KKXXXXXXXXKKXXXXXXXXXXXXKKXXXXXXXXNNNXXXXXXXKKKXXXXXXNNNNNNNNNXXNNXXXXXXXXNNNNXXXXX0kxdc;'''',:clddoolccclcccccloxko;,,;;;;:cc:,,,;cll:,''.';lkKXXXXXNNNNNKx:,,,',;;;:ldOKXXKKXXXXKKKKKXXKKKKKKKKKK
# ccldOKKXXXXXXXKKKKXXXXXNNXXXXXXXNNNNNNNNNNNNNNXXXXXXXXNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXKxc;;:;'....;oxkkddddocccclloddxkdc;'''',,;;;;,'',:c:,'....';lxKXXNNNNNNXOc'''......;::cdOKKKKKXXKKKKKKKKKKKKKKKKK
# dddxOKXNNNNNNNNNNNNNNNNNNNNNNNXXNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXKKKKKKkc''''.....';clodollc:;;;;::cloxkdc;....'''''....',,'.......',cOXXXXXXXXXk:.........',;:cx0KKXXXXKKKKKK0000000KKKX
# xxkO0XXNWWWWWWWWWWWWWWWWWWNNNXKKKXKKKKKXXXXXNNNNNNNNNNNXXXXXXXXXXXXXKKKKKKKKKXXXXKXXXKk:.........,::;;:cc:;,''';;;;clodl:,..;lc;,,'.........,:llc;;lx0XXXKKKKKKOl'.........'';:x0XNNNNXXXXXKKKKKXXKKXXKX
# 0000KXNNWWWWWWWWWWWWWWWWWWWNNXKKKK00KXXNNNNNWWNNNNWWWNNNNNNNNNNNNNNNXXXNNNNNNNNNNNNNNN0o,..........''',;,,,'..',;;;;:clc::,;xKK0kkxo:,...,:ox0KKKOOKXXXNNXXXXXXKk;..........';o0XNNNNNNNNNNXNNNNXXXXXXKK
# XXXNNNNWWWWWWWWWWWWNNWWWWWWWNNNNNNNNNWWWWWWWWWWNNNNNNNNNNNNNNNNNNNNXKKKKK00000KKKKKKXXX0xc;coxxdlc,...''..',;;:lxkdc;;:cokkxONNNNNNXkc,:ok0KXXNNNNNNNNNNXNNNNNNN0o'.,:c::cc:oOKXXXXXXXXXKKKKKKXXXXK00000
# NNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXNNNXXNNNNNXXNXXXXKKKXXXXKKKXXXXXK0OxolllodooddooddddoodddxddxkO0000kocccc;.'lO0KXXNKo:::lOXXKKXNNXXXXKkdkKXXKKKXXXXXXXXXKKKXXXXXXKkolxOkllxkxOKKKKK0000000OOOO00KKK00000K
# NXXXNNXXXXKKXXXNNNNNNNNNNXXXXXXNNNXXXXXXXXXXXXXXXXXXXXXKKKXXKOxolllccccclccclllllloc:cccccccllloodddddddollxKXXNNNXkkkdxKNNNNNNNXXXXXXKKXXXK00KKK0OOO00OOO0KK0KKKKKKXXKkkO000K00KK000KKKK0OO0000K0000KKX
# XXXXNXXXXXXXXXXXNNNNNNNNWWNNNNNNNNNNNNNNNNNNXXXNNNXKKKXKKK0koc:::::ccccc::;:cccccccc::::;;;;:::::ccc:::ccllooooxk000XXK0XNNNNNNNNNNNXXXNNNNXNNNNX0OkxkOkkkOKKKKXXXNNNNNNXXXXXXKKXXKKKKXXXK00KKKKK0000KXX
# XXXXNXKKKXXXXXXXXXKKXXNWWWWWNNWNNXNXKKXXNNNNNK000kdllooooooc;;;;;;;;;;;;;:::;:::::cccc;;;;;,,;,,;;::c:;;;::;;:::cccldxkO0XXXNXXXXNNXXXXXXXXXXNNNXK00O00000KXXXXNNNNNNNNNNNNNNNNNXXXXXXXXXXKKXXXXKK00KKKK
# XXNNNNXXXXNXXNNNNNNNNNWWWWWWWWWWNNX0xOXXXXKOkoc:;,,'',,,,;;,'',,,,;;;;;,;;ll;;;::;;:cc:;,;,,,,,,,;:::;;;,,,;;:::;;;;;;;:clloddxxkO00KKKKKXXXXXXXXXXNNXKKKKKXNNNNNNNNNNWWWNNNNNNNNNXNNNXXXXXXXXXXKKKKKKXX
# NNWNNNNNNNWWWWWWWWWWWWWWWWWWWWWWWWXxd0NNXkl;,''',',,,;;;,::,,,'''''',,,,,,:oc;;;:;;;:c:;,,''',,,',:::::;,,,',;,,,,',,,,;;;;;;;::cccloooodxxkk0KXNNNNNNXXXXXXNNNNNNWWWWWNNNNXXNNNNNXXXXXXXXXXXXXKKKKKKKKK
# WWWWWWWWWWWWWWWWWWWWNNNNWWWWWWWWWN0oo0NKx:'..'''''..',,,,;::::;,'......',,;oo;,;;;:;:cc:;,'',,.''',:::c:;;,'''',,''''.';;;,,,,,;;::cc:::::::clodxkkkkOOOO000XXKKXXNNNXXXNXXKKKXXXXXKKKKKKKK00OO0000000K0
# WWWWWWWWWWWWWWWWWWWNNNNNNNNXXNNNNNkclOXk;...........',''',,,,;;,'.......',:dd;';;;:;;:::;,',''....';::::;;;;;,',,'',''',;;;,'',,;;;::::;;;;;;;::cccccccclllldkO0KKK00KKKXXXKKKKKKXXKKKKXXKKK0000KKK0KXXX
# NNWWWWNNNNWWWWNNNNNNNNNNNNXXXNNNNXxcldOo'...........,,'',,;;,,;:;,......':lxo,',;::;;:;::,.........,;,,,,::::;;;;;,'''',''''..',,,;;:;;,,;;;;;;,,''',;:cccc:::ldkO00KXXXXXKKKK0KKKK00XXXXXXXXKKXXXXXXXXX
# XXXNNNXXXXNNNXXXXXXNNNNNNNNNNNNNNNkclooc'..........',''',,,,,',;::'...';codxl,',;:;;cc::;'....'.'..'''''',;;::,;;;,'.''''......,;;,,,,'',;;::;,'....',:cc:cc:;::clodkO0KXXKKKK0KXXKKKNNNXKXXXKXXXXXXXXXX
# NXXXXXXKKXXXK0O00KXXNNNNNNNNNNNNXX0dcllc:,..........,,,,,,,,,,,,;:lolooooodo:,,,,,,;cc::;'...''.''',,''''',,,;,,;,,'..''''....'',;;;,'',,,,,,'......',;::::cc::::::cclloxO0KXXKKXXXXXXXXKKKKXXXXXKKXXXKK
# NX000000KXXXK0OkO0KXNNNXXXXXXXXXKKX0dccc:'...........,,,,,,;;:::::lddolllll:;,;;;,',:ccc;'.'.''..''',''','''',,,,,''....'''.'',,,,;;;,,'''''.......''',;::;::::::;;:cclclloxOKKKKXXXXKKKK0KKKKXXXKKKKK00
# XK0O00000KKKKKK0O0KXXXXXXXXXKKKKKXXNX0xc,'...........,;,,',;:cccc:;;;;,,,,::;,::,,,,;:cc;'.''.'..','',''''.'''''.''''''''''.'',,,,,;,,''''''.....',,'',;;;;;;;::;;;;:cclllllldOKKKKKK00000KKK0O0000000Ok
# KKKKKXKKXXKKKKKKKKKKXXXXXXXXXXXKKXXXKkc'...........''',,,'',,;;col;..  ...';;;::,,;,,,;:;...''.'''',,,''''.''''''''',,,;;,''',,,,,,,'''''...'',,,,;,..',,;;;:::::;,;;:ccloooold00000KKKKK0KK0OkkOOOOOOOk
# XXXXNNXNNNXXNNXXXXXXXXNNXNNNNXXXXXXXk;.............',,',,...;:;;c;.   ....';;;;,.',,,',;,'......''',,;,'','','',;;,,,;;:::;,,,;;,,,,,'''''..';;:::;'..',;;;::cc:;,,,;:cclooooolxKKKKKXNXXXXXXKK000O000OO
# NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNKl......'.......',,,,;,.,colcc;.    ..',;;;;'...''''''........'..',,'.,,,,,,;::;;;:cc:;;;,,,,,,,,;,''''',;::ccc;''',,;:::ccc:;,,,;:clloollooxKXXXXNNNNNNXXKKKK000KKKK
# NWWWWWWNNNNXXXNNNNNNNNNNNNNNNXXNNNXKx:'...''.......''',,;;;,,:lllc,.  ...',,,;,......'.....'.........',,'',,;::::::::::c:;;;;;,,,;;;;;,,,,,;:clllc,''',;;:::cllc:;;,;;::clllcclxKXXXXXXXXNNXXKKKKKKKK0KX
# NNWWWNNNNNNXXXNNNNNNNNNNNNXNNNNNNNNNXK0kxxc'........'''',::;;ccccc;,'....',;;;'. .............'.......,;,',;;:c::;:::;;:::;,,;;,,;;;;;;;;;;:clool:,''',;::;:cllllc;;;;;;:cc:::cdKXKXKK0KKXXXKK0KKKKKKKKX
# XNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXNNNNNNNNNNXo'........','..,:c:::;;:;,,,''.';;;,,. .  ............''....';;,,:;:ccc;,;;;,;;;;,,;,,;;;,,;;;;;:cllooc;''.'',;;;::clllc;,,,,,,;;;;;;oOKKKKKKKKXXXXKKK0KXXXKXX
# KKXNNNNNNNNXXXNNNNXXK00KKKKXXXXXXXXXXNNNNKc........';;,'..,::;,.';;,;;,..,;,,,,.      ..........'','''',;;;::;:c::,',,,;,,,,;;,,;;;,,,;;;;:cccclc,'..''',,;;:ccc:;,,,,,;;;,,;;;lOKKXXXXXXXXXXXXKKKXXXXXK
# KKXXXNNXXXXXXXNNNNXX000KK0KXXXXKKKXXXXXXXKl'.......';;;;;';::;,'';:;,;,.';:;,,,.       .......'...,,'',,:c:::,;;;;,'''','''',;,,;;;;;;;;;,;::::::,'...''',,,,;;;,,,'',,;:;,,;;;ckKKKXXXXXNNNNXXKKKXXXXXK
# KKKXXXXXXXXKKXXXNNNXKKKKKKKXXXKKKKXXXXKKKKx,.'''...';cccc::;;,,',;:;,;,.';;;''.       .............''',;coc,:;',,,,,'...'''',,,,,,;;;;;;;;;;;:c:;;'.......'''''''''',,,;;,',;:;;o0KKKKKKXXNNXKKK00KXXKKK
# KKKKKKXKKKK0KKXXXXNNXXKKKKKXXXKKKKKXXXXXKKOc''',,,,;;:;;cl:;,',,,;;;;;,,,,;,'..     ..... ..........'',;ldc'',.'',,''....''''',,,,,;,;;;;;;,;:::;;,............'''''',,;;,,;;;''lk0KKKKKXXXXKKK000KKKK0O
# KKKK0KKKKKK0KKXKKKXXXKKKKKXKKKKKKKKKXXXKKKKkc'.....','..;c;'.',;;,,;;;;'...'..      ..... ..........'',;lo:.................'''',',,,,,;;;;;;;;;;;'.  ...........''''',,,,,,;'.'cxO0000KKKKKKKKKKK00K0OO
# KK0K00KKK0000KKKK0KKKKKK00KK0KKKKKKKKXXKKKKKx;''.. .,c::c:,'',;:;',;;;,.......       .......''..  .'',,;ll;....................''''',,,,;;;,;;,,;;'.       ......',''',,,'''...':dO0000KK00KXKK000000OOO
# K00OOOOO00OO000KKXXK0KK00KXK0KKKKKKKKKKKK0OOko;.... ....;;',;:::,';;;,'''.....       ...............',,;cc'.......................'''',,;;;,;;,,,;'.      .......',,,,,,'''.. .':dO00000OO0KKK00OO00Okkk
# K000000O00OOO00KKXX000O00KXXKXK0KKKK0KKKKK0KK0kc'.. ....',,,;;;'.',:,'',,'...       .... ..............,cc,.......................'',,,,,,,''''....       .......'',,,,''','  .':dOKK0OOOO0KK00OOO0Okkkk
# KKKKKKK0OOOO000KXXXKK0O00KXNXXK0KKKK00KXKK0KKKKk:..  ...',;;;'....;;'.,,,..         ............... ....;:;.........................''........          .   ....'''',,''',:,. .':d0XKO0K0000KK0OOOOOOkkO
# KKK0O0000OO00KKKXNXXXXXKKKXNNXXK0KKK000KKK0000K0o'.......,;;,. ..','.'','..        ......... ...........',:,..........................     .    .;:;'.       ...',,'''''':l:...,cdOKK0KK0000KKK0OkOOOOO0
# 0KKK0O0KK0kOKXKKXNXXNNXXXXXXXXXXKKXXK00KKK00OO0KO:.....',,;,.. .......'...         ......    ............';;................            ......,lk0KK0x,.      ...''....',cdo;'';cdOKXK00O000KKKK00O00000
# 0O00OO0KK000KXXXXXXXNXXXXNXNNXXXXXNNXXKKXKKK00KX0c.....';:::'........'....          .. ...       ..','...';:.                         .....,:dOKKKKK00k;.   ...........';codl::ccox0K00OOO0000K00000K000
# 0O00OO0KK00KKXXXXXXXXXXXXNNNXXNNNNNXXXXXXXXXKKKK0o'......'cxd:'...........                      ...','....,,.   .................,......,cdOOk0K000KKKKx,.......''....',;:clooc:llxOOOOkkO000O00OO0K000O
# KKKXKKKXKKKKXXXXXNXXXXNNXNXNNNNNNNNXXNXXXXKKK00XKk:......,dK0kdc,'........                      .......  .',;:lollododdddoddoodxxo,..,cxkOOOOO00000KKKOkl.......''...''',cc;:lolldkOkO0OkO00OOkkxxO0OOkk
# XKKKXXXXXXKXXXXXXNNXXNNXXXXXNNXNNNXXXNNXK0KXKOOXNKko:;;::d0XKKK0xl;........                       ..... ..',;lxOOO0KKK00000kxxO0Oc',clx0OkkOOO0K0O0K00xxd;.'....''..'''..;od:,cxkOOOOO00O0000Okkxk0K0Okk
# KKKKXXXXXXXXXXXXXNNNXNNXNNNNNNXXXNNXNNXXK0KXX00XNXKKOkkOOKXKKKXXX0xl:;.......              ...     .... ..'',:okO00KXXXK0KK000K00dcoddOK0OOK0O0K0OO00Okkko:;'..':;''''...:dxl';xOO0K0KXXXXXKKK0OO0XXKK0O
# KKXXXXXXXXXXXXXKXNNNNXXNNNNNNXXXXNNNXXXXKKKXNXKXNXXXXXNXXXXKKKXXNXX00x:,'... ......         ..          ..''';lk000KXXKK0KXKXXK0K0kkxk0XXKKXK00KO00OOO000kllc'.;o:'.',..,lxxdloxdkKKKXXXNXXX00K00KKXKKXK
# XXXXXXXNNXXNNNXXXXNNNXNNNNNNNNXNNNNNNNXXXXXNNNNXXXXNXXNNXXXXXXXXXXXXXKkdl:.......::...,::::,'..  ....   ....',:d0KKKKKKKKKKKKXKKXXXK00XNNXXXXKKOk0XXK00KX0ddd:,cxo,.';'',loolloolxKXXNNNNXXK00K00KKXKXXX
# XXXXXXNNNXNNNNNNNNNWNNNNNXXXNXXXXXXNNNNNNNNNNNNXXNNXNNNNXXNNXXXXXXXXXXKK0x;''...:o:..;ldxkOkdc,...:,.   ..'..:cckKKXXXXXKKKKKXKXXNNKKXXNNXXNNXXK0KXXXXXNXKOkxookOxl:dxdoc:loclolokKXNNNXK00KXXK00KXKKXXX
# XXKXXXXNNNNNNXXNNNNNNNNXXXXXXXXXKKXXXNNNNNNNNNXKKXNNNNXNXXKXXXXXXXKKXXKXXKkdddloxxl:;;;;:loodxo;,::'..  .',..;lox0KKXNNXXKKKXXKXXXXXXXNNNXXNNNXKKXXXXXXXXXK00OOOkkkkO00KOolocokOxx0XXNNKKKKXNXXKKKXXXKXX
# KXXXXKKXNNXXXNXXXXXKXNXXXXXXXXXK00XXXNNNNNNNNNXXXXNXNNXKXXKXNNXXXKKXXXKKXXXKKXXXKkdoc;,,:ddoollcc:;,,. .';;,,;lxOKXKKXXXKKKXXXKXXXXXXXNXNXXXNNNXXXXXXXXXXXXXKK00O00OO0KKK0xloxkkkk0XXXXKKXXXNNNXXXXXXXXK
# KXXXXKKKXXXKKKXXXXKKXNXNNNNNXXXXKKXXXXXXXXXNNNXXNNXXXNXKKKXNNXXXXKKXXXKKKXXKXXXNXKOxoc''lkxoloolcc:;'...:;,:lcoOKXXXXXXKKKXXXXXNNNNXXXXNNNXXXXXXXKKXXXKXXXXK00K0OOOOOKXXXXKkOOkxxk0XNXXKKKKXNNXXXXXXKKKK
# KKKXXK0KKKK000KXXKKKXXXNNNNXNNNXXKKXXXXXXXNWNNXXXNXXXNXKKXXXXXXXKKKXNXKKKK0XXXXNXXKko:,,okOocdkxoclc:'.,;,:oxxk000KKXXXKKXXXKXXXNNXXXNXXNWNNXXXXXKKKKKKKKXXKKKK0OOO0KXXXXXX0O0kddk0KNXK000KXXXXXXXXXKKKK
# 00KXXK0KK00K0KKXXXXXXXNNNNXXXNXXXXXXKXXKKKXXXXXXXNNXXXXXKXXXNNNXKXXNXXXXKKKXXXXXXX0Oxooox00xooxxdcloo:',',lkxk00KK000KK0KKXKXXXXNXXXXNXXNWWNXXXXKKKKXXKKKKKKKKXKKK0XXXXXKKXK00xoxkOKXKK000KXXXXKKXXKK0KK
# 00KXXXKKKKKKKKKXXXXXXXXNNNNXXXXNNXNXXKK0O0KXXKKXXXXXKXXXXKKXNNNXXXXXXXXXXXXXXXXXXX000kdok0kdoodxxdoollc:cddxkO0KK000000O0KXXXXXXKXXXXXXXNNNXXXXKKKKXXXKKXXKKKXXKKKXXXKKXKKKKKOxdkOOKKKKK0KKKXXXXKKK000KK
# 0KXXXXKKKXXKKKXXXXKKXNXXXXXXXNXXXXNNXKKOO0K000KK0KK00KK0KKKKNNXXXXKKXXXXXXXXXKKXXKKK0x:lkkdoddxkkxxolloodxdkO0KK0000000OOKXXXKXXKXXXXXXKXXXXXXKKKKKXXKKKKXKKKXKKKXXXXKKKKKK00OxxO0OO00KKKKXXXXXXKKK000K0
# KXXKXXKK0KXK0KXXXXKKKXXXXXXXXNXXXNNXXK0O0000KKKK0KK00000KK0KKXXXXXKKKKKKKKXXXXKXXXKKKxodOOxdxkxkOOkdloxddkkkO000OOOO00000KKXXKKKKKXXKKK0KKKKKKXKKKKXXK0KXXK0KKKKXXXKK00KK0000OOO0Oxk00KK0KKKXXKXK0000000
# KKK0KXX00KKOO0KKKXK00KXXXXNNNNNXXNNXKK0OO000KKXKKKK00O000OOOkOKKXXKKKKKKKKXXXXXKKKKK0kxk0Oxk000OOO0OkkkxkkkOO00OOOOO0000K0KXXK00KKXK00KK0KKKKKKK0KKKKKKKKKKKXXXKXXKKK0KKK000OkO00kxkOO0OO000000000O000OO
# KKKKKXK0O00OO0KKKKK00KKXXNNXXNNXXNXKKKKOO000KXXKKKKOxxkkOkdllok00KKKKKKKKKXNXXKKKKK00kxk0Ok0KKK000OOOkxkO0O0O00OO00000KKK0KKKK00KKKK00KK0KKKKKKK0000KKKKKKKXXXXKKKKKKKKKKK0Okk00OkkkxkOOO00OOkkO0OOOOOOO
# KKKKKK0OkO0OO0KXKKK00KKKXNXXXXXXXXXKKKK0000O0XXKKK0koldxOkxdoxO0O00KKKK0KXXXNXK00KK000000OO0K0000Ok000OOO000OO000OO00KKKKKKXXX00KKKK0KK00KKKXXKKKK00KKK0KKKKXXXK0KKKKK000K0OkO0OOkOOOO000K0OOkkOOkkkxkOO
# 000K0KK0k0KOOOKXKK00K00KXNXXXKKKKKK0KKK0O0Ok0KXX0OkkdodkO000OOKK00KKKKKKKXXXXXXK0XXK00KK00KKK000Oxk0KK0O00KKOO0K0OO000KKKKXXXX0O0KXXKKK00KKKXXK0K00000000KKKKKKK0KKKKK00000OOO0OOOOOOO0K00OkOOkOOkkxxkOk

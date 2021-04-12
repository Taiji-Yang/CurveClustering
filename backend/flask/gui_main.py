from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import datetime
from high_dataset import *
from tkinter import filedialog
canvas = None
toolbar = None
history_count = 0
user_add_file_count = 0
def plot():
	global canvas
	global toolbar
	global history_count
	history_count += 1
	if canvas:
		canvas.get_tk_widget().pack_forget()
	if toolbar:
		toolbar.pack_forget()

	fig = Figure()

	canvas = FigureCanvasTkAgg(fig, master = frame1)   
	canvas.draw() 
	ax = fig.add_subplot(111, projection="3d")
	color = ['c','m','y','g','r','b','k']
	curves, results = gui_callee(int(nc_input.get()), cmvalue_list[cmvalue - 1], affvalue_list[affvalue - 4], ifalign_list[ifalign - 10], features, history_count)
	for i in range(0, len(curves)):
		if len(curves[i]) == 3:
			ax.plot(curves[i][0], curves[i][1], curves[i][2], color = color[list(set(results)).index((results[i]))])
		elif len(curves[i]) == 2:
			ax.plot(curves[i][0], curves[i][1], color = color[list(set(results)).index((results[i]))])
		elif len(curves[i]) == 1:
			ax.plot(curves[i][0], [0]*len(curves[i][0]), color = color[list(set(results)).index((results[i]))])

	canvas.get_tk_widget().pack(anchor = NW, side = TOP, fill = BOTH, expand = True) 
	toolbar = NavigationToolbar2Tk(canvas, frame1) 
	toolbar.update() 
	log.insert(0, '---------------------------------')
	currentDT = datetime.datetime.now()
	log.insert(0, currentDT.strftime("%Y-%m-%d %H:%M:%S"))
	log.insert(0, 'success!')
	log.insert(0, 'features: ' + str(features[0]) + ', ' + str(features[1]) + ', ' + str(features[2]) )
	log.insert(0, 'if align: ' + str(ifalign_list[ifalign - 10]))
	log.insert(0, 'affinity: ' + str(affvalue_list[affvalue - 4]))
	log.insert(0, 'cmvalue: ' + str(cmvalue_list[cmvalue - 1]))
	log.insert(0, 'number of clusters: ' + str(nc_input.get()))
	log.insert(0, '---------------------------------')

def action(n):
	print(n)
	print("entry:")
	print(nc_input.get())

def set_cmvalue(n):
	global cmvalue
	cmvalue = n

def set_affvalue(n):
	global affvalue
	global align
	affvalue = n
	if affvalue not in [4,5,6]:
		v3.set(None)
		yes_align["state"] = "disabled"
		no_align["state"] = "disabled"
		align = 12
	else:
		yes_align["state"] = "active"
		no_align["state"] = "active"

def set_feature1(n,s):
	features[n] = s
	if n == 0:
		mb1['text'] = s
	elif n == 1:
		mb2['text'] = s
	elif n == 2:
		mb3['text'] = s

def set_ifalign(n):
	global ifalign
	ifalign = n

def upload_file():
	global user_add_file_count
	filename = filedialog.askopenfilename()
	source = filename
	destination = 'gui_cache/curves/user' + str(user_add_file_count) + '.csv'
	shutil.copyfile(source, destination)
	user_add_file_count += 1

cmvalue_list = ['single', 'average', 'complete', 'None']
affvalue_list = ['global_center', 'local_center', 'velocity_center', 'Euclidean', 'Frechet', 'Hausdorff', 'None']
ifalign_list = ['yes', 'no', 'None']

cmvalue = len(cmvalue_list) - 1 + 1
affvalue = len(affvalue_list) - 1 + 4
ifalign = len(ifalign_list) - 1 + 10
features = [None, None, None]

window = Tk()
window.title('Curve Clustering Library')
window.bind("<1>", lambda event: event.widget.focus_set())
wwidth = 0.8 * window.winfo_screenwidth()
wheight = 0.8 * window.winfo_screenheight()
w_x = (window.winfo_screenwidth() - wwidth)/2
w_y = (window.winfo_screenheight() - wheight)/2
window.geometry("%dx%d+%d+%d" % (wwidth,wheight,w_x,w_y))

frame1 = Frame(window)
frame2 = Frame(window)
frame1.place(relx=0, rely=0, relwidth=0.65, relheight=1)
frame2.place(relx=0.65, rely=0, relwidth=0.35, relheight=1)
'''
test_button = Button(master = frame2, command = plot, height = 3, width = 20, text = "plot")
test_button.pack(anchor = NW, side = TOP, expand = True)
'''
label_1 = Label(frame2, text = "number of clusters:", font=("Arial", 10), anchor='w')
label_1.place(relx=0.1, rely=0, relwidth=0.3, relheight=0.05)

nc_input = Entry(frame2)
nc_input.place(relx=0.4, rely=0, relwidth=0.3, relheight=0.05)

label_2 = Label(frame2, text = "clustering method:", font=("Arial", 10), anchor='w')
label_2.place(relx=0.1, rely=0.05, relwidth=0.3, relheight=0.05)

v = IntVar()
cm_single = Radiobutton(frame2, text='single', anchor=W, variable=v, value=1, command = lambda: set_cmvalue(1))
cm_single.place(relx=0.4, rely=0.05, relwidth=0.2, relheight=0.05)
cm_average = Radiobutton(frame2, text='average', anchor=W, variable=v, value=2, command = lambda: set_cmvalue(2))
cm_average.place(relx=0.6, rely=0.05, relwidth=0.2, relheight=0.05)
cm_complete = Radiobutton(frame2, text='complete', anchor=W, variable=v, value=3, command = lambda: set_cmvalue(3))
cm_complete.place(relx=0.8, rely=0.05, relwidth=0.2, relheight=0.05)

label_3 = Label(frame2, text = "affinity:", font=("Arial", 10), anchor='w')
label_3.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.05)

v2 = IntVar()
a_global_center = Radiobutton(frame2, text='global_center', anchor=W, variable=v2, value=4, command = lambda: set_affvalue(4))
a_global_center.place(relx=0.4, rely=0.1, relwidth=0.2, relheight=0.05)
a_local_center = Radiobutton(frame2, text='local_center', anchor=W, variable=v2, value=5, command = lambda: set_affvalue(5))
a_local_center.place(relx=0.6, rely=0.1, relwidth=0.2, relheight=0.05)
a_velocity_center = Radiobutton(frame2, text='velocity_center', anchor=W, variable=v2, value=6, command = lambda: set_affvalue(6))
a_velocity_center.place(relx=0.8, rely=0.1, relwidth=0.2, relheight=0.05)
a_Euclidean = Radiobutton(frame2, text='Euclidean', anchor=W, variable=v2, value=7, command = lambda: set_affvalue(7))
a_Euclidean.place(relx=0.4, rely=0.15, relwidth=0.2, relheight=0.05)
a_Frechet = Radiobutton(frame2, text='Frechet', anchor=W, variable=v2, value=8, command = lambda: set_affvalue(8))
a_Frechet.place(relx=0.6, rely=0.15, relwidth=0.2, relheight=0.05)
a_Hausdorff = Radiobutton(frame2, text='Hausdorff', anchor=W, variable=v2, value=9, command = lambda: set_affvalue(9))
a_Hausdorff.place(relx=0.8, rely=0.15, relwidth=0.2, relheight=0.05)

label_4 = Label(frame2, text = "align signal:", font=("Arial", 10), anchor='w')
label_4.place(relx=0.1, rely=0.2, relwidth=0.3, relheight=0.05)

v3 = IntVar()
yes_align = Radiobutton(frame2, text='yes', anchor=W, variable=v3, value=10, command = lambda: set_ifalign(10))
no_align = Radiobutton(frame2, text='no', anchor=W, variable=v3, value=11, command = lambda: set_ifalign(11))
yes_align.place(relx=0.4, rely=0.2, relwidth=0.2, relheight=0.05)
no_align.place(relx=0.6, rely=0.2, relwidth=0.2, relheight=0.05)

if affvalue not in [4,5,6]:
	yes_align["state"] = "disabled"
	no_align["state"] = "disabled"

label_5 = Label(frame2, text = "features", font=("Arial", 10), anchor='w')
label_5.place(relx=0.1, rely=0.25, relwidth=0.3, relheight=0.05)

menulist = ['None','time','cell_pos_x','cell_pos_y','msd','msd_sem','module_ref_y','dt','traction','aflow','module_num','module_subk','module_engage_num','module_force']
mb1 = Menubutton(frame2, text = "choose one...", anchor=W)
mb1.place(relx=0.4, rely=0.25, relwidth=0.2, relheight=0.05)
mb1.menu = Menu(mb1, tearoff = 0)
mb1["menu"] = mb1.menu
mb2 = Menubutton(frame2, text = "choose one...", anchor=W)
mb2.place(relx=0.6, rely=0.25, relwidth=0.2, relheight=0.05)
mb2.menu = Menu(mb2, tearoff = 0)
mb2["menu"] = mb2.menu
mb3 = Menubutton(frame2, text = "choose one...", anchor=W)
mb3.place(relx=0.8, rely=0.25, relwidth=0.2, relheight=0.05)
mb3.menu = Menu(mb3, tearoff = 0)
mb3["menu"] = mb3.menu
for i in range(0, len(menulist)):
	mb1.menu.add_command(label = menulist[i], command = lambda i = i: set_feature1(0, menulist[i]))
	mb2.menu.add_command(label = menulist[i], command = lambda i = i: set_feature1(1, menulist[i]))
	mb3.menu.add_command(label = menulist[i], command = lambda i = i: set_feature1(2, menulist[i]))


draw_button = Button(master = frame2, command = plot, text = "Plot")
draw_button.place(relx=0.25, rely=0.35, relwidth=0.2, relheight=0.05)

add_button = Button(master = frame2, command = upload_file, text = "Add...")
add_button.place(relx=0.55, rely=0.35, relwidth=0.2, relheight=0.05)

scrollbar = Scrollbar(frame2)
scrollbar.place(relx=0.9, rely=0.43, relwidth=0.05, relheight=0.5)
log = Listbox(frame2, yscrollcommand = scrollbar.set)
log.place(relx=0.1, rely=0.43, relwidth=0.8, relheight=0.5)
scrollbar.config(command = log.yview)



label_6 = Label(frame2, text = "Curve Clustering Library for Cell Visualization", font=("Arial", 10), anchor='e')
label_6.place(relx=0.1, rely=0.95, relwidth=0.85, relheight=0.05)


window.mainloop()

curvelist = [curve_file for curve_file in os.listdir('gui_cache/curves')]
inputlist = [input_file for input_file in os.listdir('gui_cache/inputs')]
resultlist = [result_file for result_file in os.listdir('gui_cache/clustering_results')]
for curvef in curvelist:
    os.remove(os.path.join('gui_cache/curves', curvef))
for inputf in inputlist:
    os.remove(os.path.join('gui_cache/inputs', inputf))
for resultf in resultlist:
    os.remove(os.path.join('gui_cache/clustering_results', resultf))


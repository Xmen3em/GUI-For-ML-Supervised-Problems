# Import Libraries
import tkinter as tk
from tkinter import *
from tkinter import ttk
import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from tkinter.filedialog import askopenfilename
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Start the GUI
root = Tk()
root.title('GUI FOR Supervised PROBLEMS')
root.geometry('1800x750')
root.configure(bg = '#40E0D0')
my_font = ('times', 14, 'bold')

# For Display the any DataFrame in the GUI
class DataFrameViewer(tk.Tk):
    def __init__(self, dataframe, title="DataFrame Viewer"):
        tk.Tk.__init__(self)
        self.title(title)
        self.dataframe = dataframe

        self.setup_gui()

    def setup_gui(self):
        # Create a Treeview widget
        self.tree = ttk.Treeview(self, columns=list(self.dataframe.columns), show="headings")

        # Add headings to the Treeview
        for column in self.dataframe.columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, anchor="center")

        # Insert data into the Treeview
        for index, row in self.dataframe.iterrows():
            values = list(row)
            self.tree.insert("", "end", values=values)

        # Create a vertical scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)

        # Pack the Treeview and scrollbar
        self.tree.pack(expand=tk.YES, fill=tk.BOTH)
        scrollbar.pack(side="right", fill="y")

# When you push at button OPEN
def data():
    global filename
    filename = askopenfilename(initialdir = r'/home/abdelmoneim/datasets examples',title = "Select file")
    e1.insert(0, filename)
    e1.config(text = filename)

    global file
    file = pd.read_csv(filename)
    for i in file.columns:
        box1.insert(END, i)

    global inputs
    inputs = file.iloc[:, :-1].columns.to_list()

    global target
    target = file.iloc[:, [-1]].columns.to_list()

    string_col = file.select_dtypes(include = ['object', 'string']).columns.to_list()
    string_features = file[string_col].columns.to_list()
    if target[0] in string_features:
        string_features.remove(target[0])

    num_col = file.columns.to_list()
    for col in string_col:
        num_col.remove(col)

    global numerical_features
    numerical_features = file[num_col].columns.to_list()
    if target[0] in numerical_features:
        numerical_features.remove(target[0])

    global all_inputs
    all_inputs = file.drop(labels = target, axis = 1)

    global targets
    targets = StringVar()
    targets.set('targets')
    choose = ttk.Combobox(root, width = 20, textvariable = targets, font = ('times', 13, 'bold'))
    choose['values'] = (tuple(target))
    choose.place(x = 400, y = 70)

    global num_features
    num_features = StringVar()
    num_features.set('numerical_features')
    choose = ttk.Combobox(root, width = 20, textvariable = num_features, font = ('times', 13, 'bold'))
    choose['values'] = (tuple(numerical_features))
    choose.place(x = 400, y = 110)

    global cat_features
    cat_features = StringVar()
    cat_features.set('categorical_features')
    choose = ttk.Combobox(root, width = 20, textvariable = cat_features, font = ('times', 13, 'bold'))
    choose['values'] = (tuple(string_features))
    choose.place(x = 400, y = 150)

    # for plotting
    cols = file.columns

    global X_Axis
    X_Axis = StringVar()
    X_Axis.set('X-axis')
    choose = ttk.Combobox(root, width = 22, textvariable = X_Axis, font = ('times', 13, 'bold'))
    choose['values'] = (tuple(cols))
    choose.place(x = 400, y = 285)

    global Y_Axis
    Y_Axis = StringVar()
    Y_Axis.set('Y-axis')
    choose = ttk.Combobox(root, width = 22, textvariable = Y_Axis, font = ('times', 13, 'bold'))
    choose['values'] = (tuple(cols))
    choose.place(x = 400, y = 325)

    global graphtype
    graphtype = StringVar()
    graphtype.set('Graph')
    choose = ttk.Combobox(root, width = 22, textvariable = graphtype, font = ('times', 13, 'bold'))
    choose['values'] = ('scatter', 'line', 'bar', 'hist', 'corr', 'pie', 'explained variance', 'best n_neighbors')
    choose.place(x = 400, y = 365)

    # for preprocessing
    global drop_col
    drop_col = StringVar()
    drop_col.set('Dropping column')
    choose = ttk.Combobox(root, width = 22, textvariable = drop_col, font = ('times', 13, 'bold'))
    choose['values'] = (tuple(cols)) + (tuple([['None']]))
    choose.place(x = 50, y = 285)

    global miss
    miss = StringVar()
    miss.set('Handling Missing Val')
    choose = ttk.Combobox(root, width = 22, textvariable = miss, font = ('times', 13, 'bold'))
    choose['values'] = ('drop missing values', 'Mean Strategy', 'fillna with zero', 'None')
    choose.place(x = 50, y = 325)

    global categ
    categ = StringVar()
    categ.set('Handling Categorical')
    choose = ttk.Combobox(root, width = 22, textvariable = categ, font = ('times', 13, 'bold'))
    choose['values'] = ('Label Encoder', 'One Hot Encoder', 'None')
    choose.place(x = 50, y = 365)

    global scaling
    scaling = StringVar()
    scaling.set('Scale the data')
    choose = ttk.Combobox(root, width = 22, textvariable = scaling, font = ('times', 13, 'bold'))
    choose['values'] = ('Standard Scaler', 'MinMax Scaler', 'None')
    choose.place(x = 50, y = 405)
    
    global choose_num_features
    choose_num_features = StringVar()
    choose_num_features.set('')
    Label(root, text = 'PCA_Components',  bg = 'black', font = my_font, fg = 'white').place(x = 50, y = 445)
    choose = ttk.Entry(root, width = 15, text = '', textvariable = choose_num_features, font = ('times', 13, 'bold'))
    choose.place(x = 50, y = 485)

    global decompose
    decompose = StringVar()
    decompose.set('Principle Component Analysis')
    choose = ttk.Combobox(root, width = 22, textvariable = decompose, font = ('times', 13, 'bold'))
    choose['values'] = ('PCA', 'None')
    choose.place(x = 50, y = 525)

    global choose_num_to_select
    choose_num_to_select = StringVar()
    choose_num_to_select.set('')
    Label(root, text = 'num_features_to_select', bg = 'black', font = my_font, fg = 'white').place(x = 50, y = 565)
    choose = ttk.Entry(root, width = 15, text = '', textvariable = choose_num_to_select, font = ('times', 13, 'bold'))
    choose.place(x = 50, y = 605)
    
    global selection
    selection = StringVar()
    selection.set('Feature Selection')
    choose = ttk.Combobox(root, width = 22, textvariable = selection, font = ('times', 13, 'bold'))
    choose['values'] = ('RFE', 'None')
    choose.place(x = 50, y = 645)

    global sampling
    sampling = StringVar()
    sampling.set('Over Sampling')
    choose = ttk.Combobox(root, width = 22, textvariable = sampling, font = ('times', 13, 'bold'))
    choose['values'] = ('SMOTE', 'None')
    choose.place(x = 50, y = 685)

# when you push at button show Data
def show():
    global filename
    raw_data = pd.read_csv(filename)
    DataFrameViewer(raw_data, title = 'Display The Data')

# when you want to show the missing values of your data
def missed():
    global filename
    raw_data = pd.read_csv(filename)
    mess = '{}'.format(raw_data.isnull().sum())
    messege_window = tk.Toplevel(root)
    messege_window.title('Missing Values')
    messege_label = tk.Label(messege_window, text = mess)
    messege_label.pack(padx = 40, pady = 40)

# when you push at button PLOT
def plot():
    global X_Axis
    global Y_Axis
    global graphtype
    global inputs
    global target, numerical_features
    global file
    global all_inputs
    global x_train, y_train, x_test, y_test
    
    fig = Figure(figsize = (6, 6), dpi = 70)
    u = graphtype.get()

    if  u == 'scatter':
        plot_1 = fig.add_subplot(111)
        plt.scatter(file[X_Axis.get()], file[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()
    
    if u == 'line':
        plot_1 = fig.add_subplot(111)
        plt.plot(file[X_Axis.get()], file[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u=='bar':
        plot1 = fig.add_subplot(111)
        plt.bar(file[X_Axis.get()], file[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u=='hist':
        plot1 = fig.add_subplot(111)
        plt.hist(file[X_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u=='corr':
        a = target[0]
        another_num = numerical_features.copy()
        another_num.append(a)
        plot1 = fig.add_subplot(111)
        sns.heatmap(file[another_num].corr(), annot = True)
        plt.xticks(rotation = 60)
        plt.show()

    if u=='pie':
        plot1 = fig.add_subplot(111)
        plt.pie(file[Y_Axis.get()].value_counts(), labels = file[Y_Axis.get()].unique())
        plt.show()
    
    if u == 'explained variance':
        plot1 = fig.add_subplot(111)
        pca = PCA()
        pca.fit_transform(all_inputs)
        exvar = pca.explained_variance_ratio_
        cexvarsum = np.cumsum(exvar)
        plt.bar(range(0,len(exvar)), exvar, label='Individual explained variance')
        plt.step(range(0,len(cexvarsum)), cexvarsum ,label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='lower right')
        plt.show()
    
    if u == 'best n_neighbors':
        plot1 = fig.add_subplot(111)
        mean_acc = np.zeros(20)
        for i in range(1,21):
            knn = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
            y_hat = knn.predict(x_test)
            mean_acc[i-1] = accuracy_score(y_test, y_hat)
        plt.plot(range(1,21),mean_acc)
        plt.xticks(np.arange(1,21,1.0))
        plt.xlabel('number of neighbors')
        plt.ylabel('Accuracy')
        plt.show()

# when you wnat to apply any processing step
def Preprocess():
    global drop_col
    global miss
    global categ
    global scaling
    global selection
    global sampling
    global decompose
    global num_features
    global cat_features
    global targets
    global numerical_features
    global file
    global inputs
    global target
    global all_inputs
    global choose_num_features
    global choose_num_to_select
    global x, y
    global x_train, x_test, y_train, y_test

    x = all_inputs
    y = file[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = float(split.get()) ,random_state = 42)
    
    # for dropping any column you want
    dr = drop_col.get()
    if dr == 'None' or dr == 'Dropping column':
        pass

    else:
        file = file.drop([dr], axis = 1)
        all_inputs = file.drop(labels = target, axis = 1)

        if dr in numerical_features:
            numerical_features.remove(dr)

        mess = '{} column dropped'.format(dr)
        messege_window = tk.Toplevel(root)
        messege_window.title('Missing Values')
        messege_label = tk.Label(messege_window, text = mess)
        messege_label.pack(padx = 40, pady = 40)

    # for handling any missing values
    m = miss.get()
    if m == 'drop missing values':
        mess = '{} values dropped'.format(file.isnull().sum().sum())
        file.dropna(inplace = True)
        all_inputs = file.drop(labels = target, axis = 1)
        messege_window = tk.Toplevel(root)
        messege_window.title('Missing Values')
        messege_label = tk.Label(messege_window, text = mess)
        messege_label.pack(padx = 40, pady = 40)

    elif m == 'fillna with zero':
        mess = '{} of {} values replaced with zero value'.format(all_inputs[num_features.get()].isnull().sum(), num_features.get())
        impute  = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0)
        impute.fit(all_inputs[[num_features.get()]])
        all_inputs[[num_features.get()]] = impute.transform(all_inputs[[num_features.get()]]) 
        messege_window = tk.Toplevel(root)
        messege_window.title('Missing Values')
        messege_label = tk.Label(messege_window, text = mess)
        messege_label.pack(padx = 40, pady = 40)

    elif m == 'Mean Strategy':
        mess = '{} of {} values replaced with mean value'.format(all_inputs[num_features.get()].isnull().sum(), num_features.get())
        impute  = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        impute.fit(all_inputs[[num_features.get()]])
        all_inputs[[num_features.get()]] = impute.transform(all_inputs[[num_features.get()]])
        messege_window = tk.Toplevel(root)
        messege_window.title('Missing Values')
        messege_label = tk.Label(messege_window, text = mess)
        messege_label.pack(padx = 40, pady = 40)

    elif m == 'None':
        pass

    # for handling any categorical data
    c = categ.get()
    if c == 'Label Encoder':

        if targets.get() == 'targets':
            le = LabelEncoder()
            all_inputs[cat_features.get()] = le.fit_transform(all_inputs[cat_features.get()])
            mess = '{}'.format(all_inputs[cat_features.get()].value_counts())
            messege_window = tk.Toplevel(root)
            messege_window.title('Label Encoder of {}'.format(cat_features.get()))
            messege_label = tk.Label(messege_window, text = mess)
            messege_label.pack(padx = 40, pady = 40)

        else:
            le = LabelEncoder()
            file[targets.get()] = le.fit_transform(file[targets.get()])
            mess = '{}'.format(file[targets.get()].value_counts())
            messege_window = tk.Toplevel(root)
            messege_window.title('Label Encoder of {}'.format(targets.get()))
            messege_label = tk.Label(messege_window, text = mess)
            messege_label.pack(padx = 40, pady = 40)
        
    elif c == 'One Hot Encoder':
        all_inputs = pd.get_dummies(data = all_inputs, columns = [cat_features.get()])
        DataFrameViewer(all_inputs.head(20), title = 'One Hot Encoder of {}'.format(cat_features.get()))

    elif c == 'None':
        pass

    # for scalling the data
    s = scaling.get()
    if s == 'Standard Scaler':
        scale = StandardScaler()
        scale.fit(all_inputs[numerical_features])
        all_inputs[numerical_features] = scale.transform(all_inputs[numerical_features])
        DataFrameViewer(all_inputs.head(20), title = 'Scaling with {}'.format(s))

    elif s == 'MinMax Scaler':
        min_max = MinMaxScaler()
        min_max.fit(all_inputs[numerical_features])
        all_inputs[numerical_features] = min_max.transform(all_inputs[numerical_features])
        DataFrameViewer(all_inputs.head(20), title = 'Scaling with {}'.format(s))

    elif s == 'None':
        pass

    # for apply Principle Component Analysis
    d = decompose.get()
    ch = choose_num_features.get()
    fig = Figure(figsize = (6, 6), dpi = 70)
    if d == 'PCA':
        pca = PCA(n_components = int(ch))
        pca.fit_transform(all_inputs)
        plot_1 = fig.add_subplot(111)
        pca_variance = pca.explained_variance_ratio_
        plt.bar(range(int(ch)), pca_variance, alpha = 0.5, align = 'center', label = 'individual Variance')
        plt.legend()
        plt.show()

    # for apply Recursive Feature Elimination
    se = selection.get()
    cho = choose_num_to_select.get()
    if se == 'RFE':
        name_model = ch_model.get()

        if name_model == 'SVC':
            model = SVC(kernel = 'linear', random_state = 42)

        elif name_model == 'SVR':
            model = SVR(kernel = 'linear', random_state = 42)

        elif name_model == 'Decision Tree Classification':
            model = DecisionTreeClassifier(random_state = 42)

        elif name_model == 'Decision Tree Regression':
            model = DecisionTreeRegressor(random_state = 42)

        elif name_model == 'Random Forest classification':
            model = RandomForestClassifier(random_state = 42)

        elif name_model == 'Random Forest Regression':
            model = RandomForestRegressor(random_state = 0)

        elif name_model == 'Linear Regression':
            model = LinearRegression()

        rfe = RFE(model, n_features_to_select = int(cho))
        rfe.fit(x_train, y_train)
        filter = rfe.support_
        all_inputs = all_inputs[x[x.columns[filter]].columns]
        DataFrameViewer(all_inputs.head(20), title = 'Feature Selection (RFE)')

    elif se == 'None':
        pass

    # for apply Over Sampling
    sm = sampling.get()
    if sm == 'SMOTE':
        x = all_inputs
        y = file[target]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = float(split.get()) ,random_state = 42)
        smote = SMOTE()
        x_tr, y_tr = smote.fit_resample(x_train, y_train)
        mess = 'Before Applying SMOTE: {}\n \n After Applying SMOTE: {}'.format(y_train.value_counts(), y_tr.value_counts())
        x_train = x_tr
        y_train = y_tr
        messege_window = tk.Toplevel(root)
        messege_window.title('Over Sampling (SMOTE)')
        messege_label = tk.Label(messege_window, text = mess)
        messege_label.pack(padx = 40, pady = 40)

    elif sm == 'None':
        pass

# for apply any Model you want
def model():
    global file
    global all_inputs
    global target
    global x,y
    global Accuracy, cm, Cr
    global x_train, x_test, y_train, y_test

    x = all_inputs
    y = file[target]
    
    type_model = m_type.get()
    train_model = training.get()
    name_model = ch_model.get()
    kernels = kernel.get()

    if type_model == 'Classification':

        if name_model == 'Logistic Regression':

            if train_model == 'K_fold':
                kfold = KFold(n_splits = 5, random_state = None, shuffle = True)
                model = LogisticRegression(random_state = 42)
                model.fit(x_train, y_train)
                score = cross_val_score(model, x_test, y_test, cv = kfold)
                mess = 'Cross Validation Score: {}'.format(score.mean())
                messege_window = tk.Toplevel(root)
                messege_window.title('Cross Validation Score Score')
                messege_label = tk.Label(messege_window, text = mess)
                messege_label.pack(padx = 40, pady = 40)
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))

            elif train_model == 'train_test_split':
                model = LogisticRegression(random_state = 42)
                model.fit(x_train, y_train)
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))            

        elif name_model == 'SVC':

            if train_model == 'K_fold':
                kfold = KFold(n_splits = 5, random_state = None, shuffle = True)
                model = SVC(kernel = kernels, random_state = 42)
                model.fit(x_train, y_train)
                score = cross_val_score(model, x_test, y_test, cv = kfold)
                mess = 'Cross Validation Score: {}'.format(score.mean())
                messege_window = tk.Toplevel(root)
                messege_window.title('Cross Validation Score Score')
                messege_label = tk.Label(messege_window, text = mess)
                messege_label.pack(padx = 40, pady = 40)
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))

            elif train_model == 'train_test_split':
                model = SVC(kernel = kernels, random_state = 42)
                model.fit(x_train, y_train)
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))

        elif name_model == 'K-NN':

            if train_model == 'K_fold':
                kfold = KFold(n_splits = 5, random_state = None, shuffle = True)
                model = KNeighborsClassifier(n_neighbors = int(n_neighbor.get()))
                model.fit(x_train, y_train)
                score = cross_val_score(model, x_test, y_test, cv = kfold)
                mess = 'Cross Validation Score: {}'.format(score.mean())
                messege_window = tk.Toplevel(root)
                messege_window.title('Cross Validation Score Score')
                messege_label = tk.Label(messege_window, text = mess)
                messege_label.pack(padx = 40, pady = 40)           
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))

            elif train_model == 'train_test_split':
                model = KNeighborsClassifier(n_neighbors = int(n_neighbor.get()))
                model.fit(x_train, y_train)
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))

        elif name_model == 'Decision Tree Classification':

            if train_model == 'K_fold':
                kfold = KFold(n_splits = 5, random_state = None, shuffle = True)
                model = DecisionTreeClassifier(criterion = criter.get(), random_state = 42)
                model.fit(x_train, y_train)
                score = cross_val_score(model, x_test, y_test, cv = kfold)
                mess = 'Cross Validation Score: {}'.format(score.mean())
                messege_window = tk.Toplevel(root)
                messege_window.title('Cross Validation Score Score')
                messege_label = tk.Label(messege_window, text = mess)
                messege_label.pack(padx = 40, pady = 40)               
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))

            elif train_model == 'train_test_split':

                model = DecisionTreeClassifier(criterion = criter.get(), random_state = 42)
                model.fit(x_train, y_train)
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))

        elif name_model == 'Random Forest classification':

            if train_model == 'K_fold':
                kfold = KFold(n_splits = 5, random_state = None, shuffle = True)
                model = RandomForestClassifier(criterion = criter.get(), n_estimators = int(n_estimator.get()), random_state = 42)
                model.fit(x_train, y_train)
                score = cross_val_score(model, x_test, y_test, cv = kfold)
                mess = 'Cross Validation Score: {}'.format(score.mean())
                messege_window = tk.Toplevel(root)
                messege_window.title('Cross Validation Score Score')
                messege_label = tk.Label(messege_window, text = mess)
                messege_label.pack(padx = 40, pady = 40)               
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))

            elif train_model == 'train_test_split':
                model = RandomForestClassifier(criterion = criter.get(), n_estimators = int(n_estimator.get()), random_state = 42)
                model.fit(x_train, y_train)
                Accuracy = accuracy_score(y_test, model.predict(x_test))
                cm = confusion_matrix(y_test, model.predict(x_test))
                Cr = classification_report(y_test, model.predict(x_test))            

    elif type_model == 'Regression':

        if name_model == 'SVR':

            if train_model == 'K_fold':
                kfold = KFold(n_splits = 5, random_state = None, shuffle = True)
                model = SVR(kernel = kernels)
                model.fit(x_train, y_train)
                score = cross_val_score(model, x_test, y_test, cv = kfold)
                mess = 'Cross Validation Score: {}'.format(score.mean())
                messege_window = tk.Toplevel(root)
                messege_window.title('Cross Validation Score Score')
                messege_label = tk.Label(messege_window, text = mess)
                messege_label.pack(padx = 40, pady = 40)              
                Accuracy = r2_score(y_test, model.predict(x_test))

            elif train_model == 'train_test_split':
                model = SVR(kernel = kernels)
                model.fit(x_train, y_train)
                Accuracy = r2_score(y_test, model.predict(x_test))

        elif name_model == 'Linear Regression':
            x = all_inputs
            y = file[target]

            if train_model == 'K_fold':
                kfold = KFold(n_splits = 5, random_state = None, shuffle = True)
                model = LinearRegression()
                model.fit(x_train, y_train)
                score = cross_val_score(model, x_test, y_test, cv = kfold)
                mess = 'Cross Validation Score: {}'.format(score.mean())
                messege_window = tk.Toplevel(root)
                messege_window.title('Cross Validation Score Score')
                messege_label = tk.Label(messege_window, text = mess)
                messege_label.pack(padx = 40, pady = 40) 
                Accuracy = r2_score(y_test, model.predict(x_test))

            elif train_model == 'train_test_split':
                model = LinearRegression()
                model.fit(x_train, y_train)
                Accuracy = r2_score(y_test, model.predict(x_test))

        elif name_model == 'Decision Tree Regression':

            if train_model == 'K_fold':
                kfold = KFold(n_splits = 5, random_state = None, shuffle = True)
                model = DecisionTreeRegressor(random_state = 42)
                model.fit(x_train, y_train)
                score = cross_val_score(model, x_test, y_test, cv = kfold)
                mess = 'Cross Validation Score: {}'.format(score.mean())
                messege_window = tk.Toplevel(root)
                messege_window.title('Cross Validation Score Score')
                messege_label = tk.Label(messege_window, text = mess)
                messege_label.pack(padx = 40, pady = 40)
                Accuracy = r2_score(y_test, model.predict(x_test))

            elif train_model == 'train_test_split':
                model = DecisionTreeRegressor(random_state = 42)
                model.fit(x_train, y_train)
                Accuracy = r2_score(y_test, model.predict(x_test))

        elif name_model == 'Random Forest Regression':

            if train_model == 'K_fold':
                kfold = KFold(n_splits = 5, random_state = None, shuffle = True)
                model = RandomForestRegressor(n_estimators = int(n_estimator.get()), random_state = 42)
                model.fit(x_train, y_train)
                score = cross_val_score(model, x_test, y_test, cv = kfold)
                mess = 'Cross Validation Score: {}'.format(score.mean())
                messege_window = tk.Toplevel(root)
                messege_window.title('Cross Validation Score Score')
                messege_label = tk.Label(messege_window, text = mess)
                messege_label.pack(padx = 40, pady = 40)               
                Accuracy = r2_score(y_test, model.predict(x_test))

            elif train_model == 'train_test_split':
                model = RandomForestRegressor(n_estimators = int(n_estimator.get()), random_state = 42)
                model.fit(x_train, y_train)
                Accuracy = r2_score(y_test, model.predict(x_test))

# for showing the accuracy of your model
def accuracy():
    global Accuracy

    acc = 'the Accuracy of the Model is: {:0.2f}'.format(Accuracy * 100)
    messege_window = tk.Toplevel(root)
    messege_window.title('Accuracy Score')
    messege_label = tk.Label(messege_window, text = acc)
    messege_label.pack(padx = 40, pady = 40)

# for showing the confusion matrix of your model
def con_matric():
    global cm

    matrix_window = tk.Toplevel(root)
    matrix_window.title("Confusion Matrix Heatmap")
    sns.set(font_scale=1.2)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    canvas = FigureCanvasTkAgg(ax.get_figure(), master=matrix_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# for showing the classification report of your model
def class_report():
    global Cr

    report_window = tk.Toplevel(root)
    report_window.title("Classification Report")
    text_widget = tk.Text(report_window, wrap=tk.NONE, width=60, height=10)
    text_widget.insert(tk.END, Cr)
    text_widget.pack(padx=10, pady=10)


listbox = Listbox(root,selectmode="multiple")
listbox.pack

l1 = Label(root, text = 'Select Data File', bg = 'black', font = my_font, fg = 'white')
l1.grid(row = 0, column = 0)

e1 = Entry(root, text = '', font = ('times', 13, 'bold'))
e1.place(x = 170, y = 2)

Button(root, text = 'Open',command = data, bg = 'white', font = my_font, fg = 'black').place(x = 360, y = 0)

Button(root, text = 'Show Data',command = show, bg = 'white', font = my_font, fg = 'black').place(x = 450, y = 0)

Button(root, text = 'Show Missing Values',command = missed, bg = 'white', font = my_font, fg = 'black').place(x = 600, y = 0)

box1 = Listbox(root, selectmode = 'multiple', font = ('times', 13, 'bold'))
box1.place(x = 150, y = 40)

Button(root, text = 'Preprocessing',command = Preprocess, bg = 'white', font = my_font, fg = 'black').place(x = 50, y = 240)

Button(root, text = 'Plot', command = plot, bg = 'white', font = my_font, fg = 'black').place(x = 425, y = 240)

Button(root, text = 'Modelling', bg = 'white', font = my_font, fg = 'black').place(x = 1180, y = 50)

Button(root, text = 'SVM', bg = 'white', font = my_font, fg = 'black').place(x = 800, y = 110)

Button(root, text = 'K-NN', bg = 'white', font = my_font, fg = 'black').place(x = 1200, y = 110)

Button(root, text = 'Decision Tree', bg = 'white', font = my_font, fg = 'black').place(x = 1500, y = 110)

Button(root, text = 'Random Forest', bg = 'white', font = my_font, fg = 'black').place(x = 1500, y = 260)

Button(root, text = 'Run the Model',command = model, bg = 'white', font = my_font, fg = 'black').place(x = 1200, y = 390)

Button(root, text = 'Evaluation', bg = 'white', font = my_font, fg = 'black').place(x = 1225, y = 570)

Button(root, text = 'Accuracy',command = accuracy, bg = 'white', font = my_font, fg = 'black').place(x = 900, y = 620)

Button(root, text = 'Classification Report',command = class_report, bg = 'white', font = my_font, fg = 'black').place(x = 1200, y = 620)

Button(root, text = 'Confusion Matrix',command = con_matric, bg = 'white', font = my_font, fg = 'black').place(x = 1500, y = 620)

Label(root,  text = 'Kernel', bg = 'black', font = my_font, fg = 'white').place(x = 800, y = 155)
kernel = tk.StringVar()
choose = ttk.Combobox(root, width = 30, textvariable = kernel, font = ('times', 13, 'bold'))
choose['values'] = ('linear', 'poly', 'rbf', 'sigmoid')
choose.place(x = 800, y = 195)

Label(root, text = 'n_neighbors', bg = 'black', font = my_font, fg = 'white').place(x = 1200, y = 155)
n_neighbor = Entry(root, text = '', font = ('times', 13, 'bold'))
n_neighbor.place(x = 1200, y = 195)

Label(root, text = 'Criterion', bg = 'black', font = my_font, fg = 'white').place(x = 1500, y = 155)
criter = tk.StringVar()
choose = ttk.Combobox(root, width = 30, textvariable = criter, font = ('times', 13, 'bold'))
choose['values'] = ('gini', 'entropy')
choose.place(x = 1500, y = 195)

Label(root, text = 'n_estimators', bg = 'black', font = my_font, fg = 'white').place(x = 1500, y = 310)
n_estimator = Entry(root, text = '', font = ('times', 13, 'bold'))
n_estimator.place(x = 1500, y = 350)

m_type = tk.StringVar()
choose = ttk.Combobox(root, width = 30, textvariable = m_type, font = ('times', 13, 'bold'))
choose['values'] = ('Regression', 'Classification')
choose.place(x = 800, y = 480)

Label(root, text = 'Training', bg = 'black', font = my_font, fg = 'white').place(x = 1500, y = 440)
training = tk.StringVar()
choose = ttk.Combobox(root, width = 30, textvariable = training, font = ('times', 13, 'bold'))
choose['values'] = ('K_fold', 'train_test_split')
choose.place(x = 1500, y = 480)

Label(root, text = "split_size", bg = 'black', font = my_font, fg = 'white').place(x = 1200, y = 440)
split = StringVar()
split.set('0.2')
choose = ttk.Combobox(root, width = 30, textvariable = split, font = ('times', 13, 'bold'))   
choose['values'] = ('0.2', '0.25', '0.3')
choose.place(x = 1200, y = 480)

Label(root, text = "Choose the Model", bg = 'black', font = my_font, fg = 'white').place(x = 800, y = 440)
ch_model = StringVar()
choose = ttk.Combobox(root, width = 30, textvariable = ch_model, font = ('times', 13, 'bold'))
choose['values'] = ('Logistic Regression', 'SVC', 'K-NN', 'Decision Tree Classification','Random Forest classification', 'Decision Tree Regression', 'Random Forest Regression', 'Linear Regression', 'SVR')
choose.place(x = 800, y = 520)

root.mainloop()
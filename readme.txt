#########################################        Python             ########################################
Note: For AUC score predicted probability of Naive Bayes i.e "submission_AUC.csv" should be use and fro
Recall and precision "submission_Recall_precision.csv" should be used. If only one file to consider than
"submission_Recall_precision.csv" should be used as "submission_AUC.csv" is predicted probabilities and so cant
be used to calculate Recall and Precision.

Need to install all the below mentioned libraries of specific functions with their dependencies:
Python 3.6.8
Librarry os (inbuild)
Library pandas 0.24.2
Library numpy 1.16.4
Library matplotlib 3.1.1 
Library seaborn 0.9.0
Library sklearn 0.21.2
Library imblearn 0.5.0
Library scipy 1.2.1

1. There are two files "python_gui.py" which needs to be run in an IDE or a GUI like spyder and nother file is
	"python_cli.py" which needs to be run in cli window like DOS. If "python_cli.py" is choosed to be run then
	the script output should be redirected to an output file (python __directory of py file__ > outputfile.txt) in order to get
	proper understanding of the output. The cli python file doesnot include any visualization code. If visualisation is
	needed the only "python_gui.py" file should be run in a GUI like spyder. Kindly Run both the files where the dataset "train.csv"
	and "test.csv" are present with the same file name
NOTE:- Kindly change the working directory in both the code files as required.

#########################################        R            ########################################
Need to install all the below mentioned libraries of specific functions with their dependencies:
R version 3.5.1
Librarry ROSE_0.0-3 
Librarry DMwR_0.4.1 
Librarry pROC_1.12.1
Librarry class_7.3-15 
Librarry randomForest_4.6-14
Librarry rpart_4.1-15 
Librarry e1071_1.7-2
Librarry caret_6.0-80 
Librarry lattice_0.20-38     
Librarry gridExtra_2.3      
Librarry dplyr_0.7.6         
Librarry ggplot2_3.0.0       
Librarry tibble_1.4.2        
Librarry data.table_1.12.2  

1. There are two files "R_gui.R" which needs to be run in an IDE or a GUI like R_studio and another file is
	"R_cli.R" which needs to be run in cli window like DOS. If "R_cli.R" is choosed to be run then
	the script output should be redirected to an output file (__R.exe directory__ __directory of R file__) or from cmd if 
	R is in enviorment variable then "R CMD BATCH script.R" in order to get	proper understanding of the output. The cli R
	file doesnot include any visualization code. If visualisation is needed the only "R_gui.R" file should be run in a GUI
	like R_studio. Kindly Run both the files where the dataset "train.csv" and "test.csv" are present with the same file name
NOTE:- Kindly change the working directory in both the code files as required.


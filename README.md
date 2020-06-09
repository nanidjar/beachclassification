# beach_classification
Automate the lidar point cloud classification process. Currently can only distinguish between water and subaerial beach.

Quickstart:

  1. Create two folders. One will store processed surveys, and the other will store trained classifiers and their metadata.
  2. Download the Anaconda distribution of Python 3. ( https://www.anaconda.com/ )
  3. Go to *C:/Users/YourUsername/.jupyter*
  4. Using a text editor, open *jupyter_notebook_config*.
   		Change 
							*#c.NotebookApp.notebook_dir = u''*
				to
							*c.NotebookApp.notebook_dir = 'D:/Your/Path'*
				This should be a folder where you will store your scripts to use this package.
  5. Open the program 'Anaconda Prompt'. Search for it in the Windows search bar.
  6. In this terminal, type 
							*>pip install git+https://github.com/nanidjar/beachclassification.git#egg=beachclassification*
  7. After it has finished installing, type 
							*jupyter notebook*
		 into the terminal. 
  8. On the jupyter notebooks homepage, in the upper right corner, click 'New' and then 'Python 3'.
  9. Into the first cell, type 
							*import beachclassification.survey as survey*
							*import beachclassification.autolabel as autolabel*
  10. In a new cell, type
							*help(survey)*
			and then
							*help(autolabel)*
			to learn about the classes, methods, and workflow for this project. 
  11. As you experiment with this package, if you get any errors or odd output, 
			please take a screenshot of your script and the error message, and send to me (Niv).
	

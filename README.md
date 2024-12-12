[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOAIG_9o)
# Coursework 2 - Deep Learning Module 2024

This repository and README contains relevant information regarding the Coursework 2 of the Deep Learning module, please read it all carefully. 

| **COURSEWORK** | **Date and time** |
| :------------------ | :---: |
| *Release* | Thursday 12 Dec 14:00h |
| *Due (deadline)* | Friday 13 Dec 18:00h |

**At 18:00h on Friday, I will clone your repositories, so make sure everything is committed by then!**


# Table of contents
1. [Assessment description](#description)
2. [Assessment format](#format)
3. [Deliverables](#deliverables)
4. [Tips and tricks](#tipsandtricks)
5. [Academic integrity](#academicintegrity)


## Assessment description <a name="description"></a>

For our second assessment, we will be looking at sequential weather data. The data contains a number of relevant weather variables, recorded daily for a European city during a period of four decades.

However, this dataset has been corrupted, and some of the daily measurements are now missing. The goal of this assessment is to develop a neural network architecture that can recover the missing measurements!

The dataset has been saved as `.csv` files and has been separated into a `training_set/` folder and a `test_set.csv` file. The `training_set/` contains data for three of the four decades, separated in one individual file per decade:

- `training_set_0.csv` contains corrupted data for the first decade; `training_set_0_nogaps.csv` contains the same data before it was corrupted.
- `training_set_1.csv` contains corrupted data for the second decade; `training_set_1_nogaps.csv` contains the same data before it was corrupted.
- `training_set_2.csv` contains corrupted data for the third decade; `training_set_2_nogaps.csv` contains the same data before it was corrupted.

The `test_set.csv` file contains the remaining decade; this data is corrupted and contains gaps, but we have lost access to the data before corruption.

The architecture that you design in this assessment should use the data contained inside the `training_set` in order to recover the missing information in the data in `test_set.csv`. Note that decades `0`, `1`, and `2` are not necessarily consecutive.

Inside each of the `.csv` files, you will find the following columns:

- `date`
- `cloud_cover`
- `sunshine`
- `global_radiation`
- `max_temp`
- `mean_temp`
- `min_temp`
- `precipitation`
- `pressure`


## Assessment format <a name="format"></a>


A notebook file called `Assessment.ipynb` is provided for you to **solve** and **turn in** the assessment. All answers to the assessment should be contained within
the `Assessment.ipynb` notebook and **the name of the notebook should NOT be changed**. 

Inside the `Assessment.ipynb` file, you will find three questions. Please, do not change the structure the notebook, but you are free to 
add new code and text cells as required to your answers. Read the text for each question and follow the instructions carefully. 
Answers that do not follow this structure will not be marked.

Please, **make sure to execute all your cells and save the result of the execution**. We will only mark cells that have been executed and will not execute any
cells ourselves.


### Question 1 (25%)

Load the training and test datasets. Then, use the space below to present the following set of figures:

1. Using line plots, show the different variables in each dataset, both before and after corruption. Choose a single decade to plot, and plot the first 365 days of data. Plot the time series for each variable in an independent axis window. Make sure the axis windows are sized appropriately so that trends and corruption in the variables can be easily observed.
2. Plot a histogram for each variable in each dataset, across all decades, both before and after corruption.

### Question 2 (25%)

Using the data loaded in **Question 1**, create a PyTorch `TensorDataset`, and create one `DataLoader` for the training set and another one for the test set.

The training loader should provide batches of weather data that have been corrupted, as well as the corresponding, paired un-corrupted batch of data. The test loader should provide batches of corrupted weather data, with no corresponding uncorrupted labels.

Using line plots, show here one batch from both the training and test datasets before and after corruption. Use different axis windows for input and label of the batch.

### Question 3 (50%)

Using the dataset created in **Question 2**, design and train an architecture to recover the missing weather values of the provided test dataset.

Using line plots, show the test weather data with the missing values filled in using a different colour.

Additionally, save the test data with the missing values filled in into a file called `test_set_nogaps.csv` inside this repository. 
This file should have the same format as the original `test_set.csv` file, with the same number of rows and columns, the same row and column ordering, and the same column headings.
**Files that do not adhere to the format or use a different file name will not be assessed.**

You have freedom to choose an architecture that you consider appropriate to solve this problem. However, you will need to train your chosen architecture as part of the assessment:
**pre-trained networks are not allowed**. 

You will be assessed by the quality of your predictions of the missing data values and additional 
marks will be given for originality in your network design choices. You should include, as part of your answer, a paragraph explaining the architecture you have chosen and any additional design 
choices and hyperparameters that have been important to build your solution. 

This is an open-book assessment and you are encouraged to use resources online, including  tools like chatGPT. However, make sure to always mention the sources for your code and ideas, 
including websites, papers, and tools like chatGPT.


## Deliverables <a name="deliverables"></a>

As described in the description above, the assessment's deliverables are:

1. The completed `Assessment.ipynb` file - The name of the file should remain unchanged. The file should contain your answers to the three assessment questions.
The internal format of the notebook should also remain unchanged, but you are free to add new code and text cells as required to your answers. The notebook's cells
should be executed and saved: we will only mark cells that are executed and will not execute cells for you.

2. A generated `test_set_nogaps.csv` file - The file will contain the test dataset with the missing values filled in by the network you have proposed and trained.
The file should follow the same format as the provided `test_set.csv` file (the same number of rows and columns, the same row and column ordering, and the same column headings).
It is important that you name your answer file `test_set_nogaps.csv`.

3. The completed `References.md` file - List in this file your references using the template provided. See the section on *Academic integrity* below.

4. Fill in the [Academic integrity declaration form](https://forms.office.com/e/jFPnNNSmb8) - Please, carefully read, fill, and submit this form in addition to your GitHub Classroom submission. See the section on *Academic integrity* below.


***IMPORTANT: When you save your notebook, make sure that all the cells are executed so that when you upload the `Assessment.ipynb` file to the repository, the output of the cells are visible***


## Tips and tricks <a name="tipsandtricks"></a>

- You have one day and a half: organise your time well and think about the tasks that you will need to do to complete the assessment before diving in. Then, build
a list of tasks, prioritise it and allocate time to each task in the list, including breaks and time to rest overnight. You can leave your networks training while you
sleep, but be careful not to burn all your compute units!

- Set up your development environment carefully: do not connect to a GPU and burn credits while you are writing code or doing other tasks that do not require
 serious compute.

- Checkpoint your models during training to avoid having to retrain from epoch 0 every time you disconnect your running environment.

- Test your submission early. In particular:
	- make sure you can save the `test_set_nogaps.csv` file correctly and upload them in the repository well before the deadline to void unpleasant last-minute surprises
	- make sure that, when you download your notebook from Colab with all the executed cells, these executed cells contain the output that you want to have visible in the final repository. If not, check that you have saved and uploaded the correct notebook.
 
 
## Academic integrity <a name="academicintegrity"></a>

This is an open-book assessment and you are encouraged to use resources online. 
However, we expect that you carefully review and understand every aspect of your submission (code, documentation, comments, etc.).
If you use AI tools, you should use them solely for learning, research, or understanding purposes rather than for directly solving assessment tasks.
We expect that all "human-readable" text in the assessment (including documentation, comments, and explanations) are written in your own words, accurately representing your personal understanding, knowledge, and opinions.
We also expect you to undertake this assessment individually and not discuss your answers with other students.

If you use any external sources outside the lecture notes, you should declare these in a `References.md` file and explained their usage in the code through comments and markdown, ensuring full transparency.
Populate the markdown file in the repository called `References.md` with links to sites you have used, papers you have read, links to your chatGPT prompts and answers, and any other resource that you have used to help you design your model.

At any point after the submission of your assessment, you may be invited to an oral examination to discuss or justify any aspect of your submission, 
and you should be prepared to explain any part of your work in detail if required.

As part of the assessment, it is **mandatory** that you fill in this [Academic integrity declaration](https://forms.office.com/e/jFPnNNSmb8).
**Assessments without a corresponding declaration will not be marked.**

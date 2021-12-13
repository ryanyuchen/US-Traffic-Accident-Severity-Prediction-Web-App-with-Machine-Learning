# CSE6242 Team 76 Project (Fall 2020)

Our course project is to help the users better analyse and visualize the US car accident dataset. The users can also predict the car severity based some features loaded from the web-app interface. 

## Team members
**Cong Wang, Fang Fu, Ruiqiang Chen, Xuebing Xiang, Yao Lu, and Yu Chen**

## Description

- “Severity Prediction” tab includes a short description about the model used for prediction of the
new data and a simple tutorial on how to use. In the second part, it allows the users to input new accident
data and choose the model to predict severity level. 

- “Explore & Visualize” tab includes interactive choropleth maps and data statistics. It provides interacitve and static data visualization to help the users better understand the car accidents in US. When you click the and the link "U.S. Accidents Dataset Feature Visualization" it will usually take about 5-10 seconds to see plot displayed on the browser as the datasets processed by backend is large, more than 1 million rows. 

- “About Dataset” tab includes a sample plot, the description and source of the datasets.

- “Team Introduction” tab includes the project description and team member information.

If you find errors with our implementation, please submit a pull request on our Github page. If you want to build-upon our work, fork us (and star us if you appreciate our work!).

## Required Dependencies

- Please install the Flask, Pandas and scikit-learn libraries in order to use our application. Relevant links are available below:

(https://pandas.pydata.org/pandas-docs/stable/install.html)

(http://flask.pocoo.org/docs/0.12/installation/)

(https://scikit-learn.org/stable/)

- An easy way to install all dependencies is run *pip3 install -r requirements.txt* once you clone the repo and change directory to the main directory of the repo/project. 

## Webapp Installation and Launch

- You can choose to download the for this repository to your system or use terminal:

```
https://github.com/chenrq2005/cse6242_team76_project.git
```

- Some files with size larger than 100 MB are not included in the Git repo, like the processed dataset (dataset_featureviz.csv files) and trained model (15wRF_limit.pickle). Please download them from [this Google shared drive](https://drive.google.com/drive/folders/11G-OWjtxEsZ6_sLuW03AQNa4mlvdiTS5). After downloading, **please save that dataset_featureviz.csv file into the app foler /static/data directory and save the 15wRF_limit.pickle files into /models directory.**

We also recommend reviewing the website on Kaggle <a href="https://www.kaggle.com/sobhanmoosavi/us-accidents/" target="_blank">here</a> that helps understand the various columns/attributes of the data.

After above operation the structure of files in the Flask based webapp should be like this:

```
├── README.md
├── app.py
├── extralib
│   ├── jquery-2.1.4.min.js
│   └── socket.io.js
├── models
│   ├── 15wRF_limit.pickle
│   ├── 15wXGB_limit.pickle
├── requirements.txt
├── static
│   ├── css
│   ├── data
│   ├── img
│   ├── js
│   └── lib
└── templates
    ├── about.html
    ├── dataset.html
    ├── explore.html
    ├── featureviz_YC.html
    ├── index.html
    ├── severitybycounty1_YC.html
    └── severitybyyear_YC.html
```

- Open terminal and change working directory to the main directory of the CSE6242_team76_project directory. **Please make sure required dependencies were correctly installed!** Run the program *app.py* in python:
```
python app.py
```
- If flask has been installed as stated above and there are no other errors (hopefully!), then you should see something like this in your terminal:
```
* Debug mode: on
* Running on http://localhost:6242/ (Press CTRL+C to quit)
* Restarting with stat
* Debugger is active!
* Debugger PIN: xxx-xxx-xxx
```
- Now open a browser of your choice (we like Firefox; Chrome, Safari distort our beautiful work), and type in http://localhost:6242/ to your search bar. You should have reached the landing page of the project. Congratulations!

## Execution

- “Severity Prediction” page: Step 1 allows the users to choose the machine learning model. It has already implanted Random Forest and XGBoost. You can input the new data in Step 2. It also provides a link to obtain the latitude and longitude of the accident location. The predicted accident severity, the picked model and input variables will be shown at the right bottom after clicking the “predict accident severity” button. 

- “Explore & Visualize” page includes interactive visualizations and data statistics to help the users to better understand the datasets and features. In the interactive choropleth map, one is for US country wide car accident count by county and severity and the other one is US country wide car accident by year and severity level. The visualization includes dropdowns to allow the user to customize the data and their preferences by severity level and year. A tooltip will display the data for each county upon mouseover. So they are easy for the users to learn the accident count and severity level in their county and compare to the nearby counties, which are useful for them to make effective measures and strategic decisions. The feature visualization includes 20 major features in the dropdown and allows the users to choose the interested feature. It will show the bar plot of accident count percentage of each item in the feature and a tooltip will display the accident count percentage of the selected item. The last “data statistics” includes several static plots of dataset statistics, including accident count by state, weather condition, year, month, day of week and hours. These plots show some interesting insights of the dataset, like the most accidents occur during going to work and getting off work which makes sense.

## Demo Video

- Youtube URL: https://youtu.be/xEfdnELbtAM

## FAQs

- If port number 6242 is used in other process\
  You need to change the port number in the line below in app.py
  
  ```
  if __name__ == '__main__':
    app.run(host='localhost', port=6242, debug=True)
  ```
  
- How to send feedback to the developers?\
  In the “Team Introduction” tab you can send email to the developer by clicking developer's name. 

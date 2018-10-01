# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 17:58:14 2018

@author: wangdi
"""
import datetime
import csv
import random
import numpy as np
 
def ReadData():
  with open(r'.\home_data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    house_info_list = []
    i = 0
    for row in readCSV:   
        if i > 0:
            date_year = int(row[1][:4])           
            price = int(row[2])
            bedrooms = int(row[3])
            bathrooms = float(row[4])
            sqft_living = int(row[5])
            sqft_lot = int(row[6])
            floors = float(row[7])
            waterfront = int(row[8])
            view = int(row[9])
            condition = int(row[10])
            sqft_above = int(row[12])
            sqft_basement = int(row[13])
            yr_built = date_year - int(row[14])
            house_info = {'date_year':date_year,
                          'price':price, 
                          'bedrooms':bedrooms,
                          'bathrooms':bathrooms,
                          'sqft_living':sqft_living,
                          'sqft_lot':sqft_lot,
                          'floors':floors,
                          'waterfront':waterfront,
                          'view':view,
                          'condition':condition,
                          'sqft_above':sqft_above,
                          'sqft_basement':sqft_basement,
                          'yr_built':yr_built}
            house_info_list.append(house_info)
        i += 1
    #print(house_info_list)
    return house_info_list

def SaveData(predict_price_list):
    with open(r'.\home_data_log.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in predict_price_list:
            writer.writerow(row)
    return 0    

def main():
    print('learning...')
    house_info_list = ReadData()  
    input_x = []
    output_y = []
    for house_info in house_info_list:
        input_x.append([house_info['bedrooms'],
                       house_info['bathrooms'],
                       house_info['sqft_living'],
                       house_info['sqft_lot'],
                       house_info['floors'],
                       house_info['view']+0.001,
                       house_info['condition']+0.001,
                       house_info['sqft_above']+0.001,
                       house_info['sqft_basement']+0.001,
                       house_info['yr_built']])
        output_y.append([house_info['price']])

    #log_x = np.log2(np.add(np.divide(input_x,1000),0.8))
    log_x = input_x
    X_tag = np.mat(np.matmul(np.transpose(log_x), log_x)).I
    X_tag = np.matmul(X_tag, np.transpose(log_x))    
    weight =np.matmul(X_tag, output_y)
    print('weight is')
    print(weight)    
    print('Testing...')
    
    sqrt_variance = 0
    i = 0
    predict_price_list = [weight.tolist()[0]]
    
    for x in input_x:
        #predict_price = np.matmul(np.log2(np.add(np.divide(x,1000),0.8)), weight) 
        predict_price = np.matmul(x, weight) 
        value_x = x
        value_x.append(predict_price.tolist()[0][0])
        predict_price_list.append(value_x)          
        sqrt_variance += (output_y[i][0]-predict_price)**2        
        i += 1
    SaveData(predict_price_list)
    print('The sqrt variance is ' + str(sqrt_variance))        
        
if __name__ == '__main__':
    main()
    
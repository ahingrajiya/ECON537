import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

class ARCorrelation:
    def __init__(self,file_path,max_lag):
        """Initalize the file path"""
        self.file_path = file_path
        self.data=None
        self.max_lag=max_lag
        
    def read_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("Data read successfully")
        except Exception as e:
            print("Error reading data: {e}")
        
    def calculate_correlation(self):
        if self.data is None:
            print("Data not read")
            return
        # gdpdata = np.log(self.data['GDPC1'].to_numpy())
        gdpdata = self.data['GDPC1'].to_numpy()
        mean = gdpdata.mean()
        denominator = np.sum((gdpdata-mean)**2)
        acf = np.zeros((self.max_lag,2))
        for j in range(1,self.max_lag):
            lag = j
            for i in range(0,gdpdata.size-lag):
                numerator = ((gdpdata[i]-mean)*(gdpdata[i+lag]-mean))
            acf[j,:]=[lag,numerator/(gdpdata.size *denominator)]
        return acf
    
    def acf_calculator(self,matrix):
        mean = matrix.mean()
        denominator = np.sum((matrix-mean)**2)
        acf = np.zeros((self.max_lag,2))
        for j in range(1,self.max_lag):
            lag = j
            for i in range(0,matrix.size-lag):
                numerator = ((matrix[i]-mean)*(matrix[i+lag]-mean))
            acf[j,:]=[lag,numerator/(matrix.size *denominator)]
        return acf
    
    def pacf_calculator(self,matrix):
        pacf = np.zeros((self.max_lag,2))
        for lag in range(1, self.max_lag + 1):
            X = np.column_stack([matrix[i:-lag+i] for i in range(lag)])
            X = sm.add_constant(X)
            model = sm.OLS(matrix[lag:], X)
            result = model.fit()
            pacf[lag-1,:]=[lag,result.params[1]]
        return pacf
    
    def ar1_model_fitting_undemeaned(self):
        if self.data is None:
            print("Data not read")
            return
        gdpdata = np.log(self.data['GDPC1'].to_numpy())
        model = sm.tsa.AutoReg(gdpdata,lags=1,trend='c')
        result = model.fit()
        print("Undemeaned Data Fit")
        print(result.summary())
        trend = result.fittedvalues
        residuals = gdpdata[1:] - trend
        return residuals
        
    def ar1_model_fitting_demeaned(self):
        if self.data is None:
            print("Data not read")
            return
        gdpdata = np.log(self.data['GDPC1'].to_numpy())
        gdpdata = gdpdata - np.mean(gdpdata)
        model = sm.tsa.AutoReg(gdpdata,lags=1,trend='c')
        result = model.fit()
        print("Demeaned Data Fit")
        print(result.summary())
        trend = result.fittedvalues
        residuals = gdpdata[1:] - trend
        return residuals
    
    def first_difference(self):
        if self.data is None:
            print("Data not read")
            return
        gdpdata = np.log(self.data['GDPC1'].to_numpy())
        gdpdata = gdpdata - np.mean(gdpdata)
        gdpdata = np.diff(gdpdata)
        first_diff_acf = self.acf_calculator(gdpdata)
        first_diff_pacf = self.pacf_calculator(gdpdata)
        return first_diff_acf,first_diff_pacf
    
    def plots(self,acf):
        self.data['Date'] = pd.to_datetime(self.data['observation_date'], format='%Y-%m-%d')
        self.data.set_index('Date', inplace=True)   
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, np.log(self.data['GDPC1']), label='Log of Real GDP')
        plt.xlabel('Year')
        plt.ylabel('Log of Real GDP')
        plt.title('Log-Level of Real GDP (1947:Q1 - Present)')
        plt.legend()
        plt.grid(True)
        plt.savefig("gdp_log_level.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.bar(acf[:,0].astype(int), acf[:,1].astype(float), color='blue', alpha=0.7)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Correlogram of Log Level Real GDP')
        plt.grid(True)
        plt.savefig("correlogram.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_acf_pacf (self, acf_matrix,pacf_matrix,name1,name2):
        plt.figure(figsize=(10, 5))
        plt.bar(acf_matrix[:,0].astype(int), acf_matrix[:,1].astype(float), color='blue', alpha=0.7)
        plt.xlabel('Lag')
        plt.ylabel('ACF')
        plt.title('Correlogram of Log Level Real GDP')
        plt.grid(True)
        plt.savefig(name1, dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.bar(pacf_matrix[:,0].astype(int), pacf_matrix[:,1].astype(float), color='blue', alpha=0.7)
        plt.xlabel('Lag')
        plt.ylabel('PACF')
        plt.title('Correlogram of Log Level Real GDP')
        plt.grid(True)
        plt.savefig(name2, dpi=300, bbox_inches='tight')
        plt.show()
        
    
if __name__ == "__main__":
    file_path = "GDPC1.csv"  # Replace with your actual file path
    calculator = ARCorrelation(file_path,100)
    calculator.read_data()
    result = calculator.calculate_correlation()
    residuals1 = calculator.ar1_model_fitting_undemeaned()
    acf1 = calculator.acf_calculator(residuals1)
    pacf1 = calculator.pacf_calculator(residuals1)
    # calculator.plot_acf_pacf(acf1,pacf1,"acf1.png","pacf1.png")
    residuals2 = calculator.ar1_model_fitting_demeaned()
    acf2 = calculator.acf_calculator(residuals2)
    pacf2 = calculator.pacf_calculator(residuals2)
    calculator.plot_acf_pacf(acf2,pacf2,"acf2.png","pacf2.png")
    
    diff_acf, diff_pacf = calculator.first_difference()
    calculator.plot_acf_pacf(diff_acf,diff_pacf,"diff_acf.png","diff_pacf.png")
    # calculator.plots(result)
    
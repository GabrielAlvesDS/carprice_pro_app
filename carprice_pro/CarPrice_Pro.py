import pickle
import numpy  as np
import pandas as pd

class CarPrice_Pro:
    
    def __init__( self ):
        self.AnoFabricacao_scaler     = pickle.load(open('parameter/AnoFabricacao_scaler.pkl', 'rb'))
        self.AnoModelo_scaler         = pickle.load(open('parameter/AnoModelo_scaler.pkl', 'rb'))
        self.KM_scaler                = pickle.load(open('parameter/KM_scaler.pkl', 'rb'))
        self.df_encoding              = pickle.load(open('parameter/df_encoding.pkl', 'rb'))
        self.Estado_scaler            = pickle.load(open('parameter/Estado_scaler.pkl', 'rb'))
        self.Cidade_scaler            = pickle.load(open('parameter/Cidade_scaler.pkl', 'rb'))
        self.Cambio_scaler            = pickle.load(open('parameter/Cambio_scaler.pkl', 'rb'))
        self.Cor_scaler               = pickle.load(open('parameter/Cor_scaler.pkl', 'rb'))
        self.UnicoDono_scaler         = pickle.load(open('parameter/UnicoDono_scaler.pkl', 'rb'))
        self.IPVAPago_scaler          = pickle.load(open('parameter/IPVAPago_scaler.pkl', 'rb'))
        self.Licenciado_scaler        = pickle.load(open('parameter/Licenciado_scaler.pkl', 'rb'))
        self.Blindado_scaler          = pickle.load(open('parameter/Blindado_scaler.pkl', 'rb'))
        self.TipoVendedor_scaler      = pickle.load(open('parameter/TipoVendedor_scaler.pkl', 'rb'))        

    def remove_outliers(self, df, column_name, lower_multiplier=1, upper_multiplier=1):
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1

        # Identificar outliers usando o critério IQR
        outliers = ~((df[column_name] < (Q1 - lower_multiplier * IQR)) | (df[column_name] > (Q3 + upper_multiplier * IQR)))

        # Filtrar os dados removendo os outliers
        df_filtered = df[outliers]

        # Resetar o índice para garantir um índice padrão
        df_filtered.reset_index(drop=True, inplace=True)

        return df_filtered
    
    def data_cleaning(self, df):
        # Rename Columns
        cols = ['Marca', 'Modelo', 'Versao', 'AnoFabricacao', 'AnoModelo', 'Estado', 'Cidade', 'KM', 'Cambio', 'Cor', 'UnicoDono', 'IPVAPago', 'Licenciado', 'Blindado', 'TipoVendedor']
        df = df[cols]

        # Filtering outliers in KM
        df = self.remove_outliers(df, 'KM')

        # Filtering AnoModelo
        df = self.remove_outliers(df, 'AnoModelo')

        # Filtering AnoFabricacao
        df = self.remove_outliers(df, 'AnoFabricacao')

        # Filtering Brands
        # Remover Marcas com menos de 1% do dataset
        df_aux = df['Marca'].value_counts(normalize = True).reset_index()
        df_aux = df_aux[df_aux['proportion'] >= 0.01]
        top_marcas = df_aux['Marca']
        df = df[df['Marca'].isin(top_marcas)]

        return df
    
    def data_preparation(self, df1):
        ## 5.1. Rescaling
        df1['AnoFabricacao'] = self.AnoFabricacao_scaler.transform(df1[['AnoFabricacao']].values.reshape(-1, 1))
        df1['AnoModelo']     = self.AnoModelo_scaler.transform(df1[['AnoModelo']].values.reshape(-1, 1))
        df1['KM']            = self.KM_scaler.transform(df1[['KM']].values.reshape(-1, 1))

        ## 5.2. Encoding

        # Mapeamento para 'Versao'
        mapeamento_versao = self.df_encoding.set_index(['Marca', 'Modelo', 'Versao'])['Versao_encoded'].to_dict()
        df1['Versao'] = df1[['Marca', 'Modelo', 'Versao']].apply(lambda x: mapeamento_versao.get(tuple(x), x[2]), axis=1)

        mapeamento_modelo = self.df_encoding.set_index(['Marca', 'Modelo'])['Modelo_encoded'].to_dict()
        df1['Modelo'] = df1[['Marca', 'Modelo']].apply(lambda x: mapeamento_modelo.get(tuple(x), x[1]), axis=1)

        mapeamento_marca = self.df_encoding.set_index('Marca')['Marca_encoded'].to_dict()
        df1['Marca'] = df1['Marca'].map(mapeamento_marca)


        df1['Estado']        = self.Estado_scaler.transform(df1['Estado'].values)
        df1['Cidade']        = self.Cidade_scaler.transform(df1['Cidade'].values)
        df1['Cambio']        = self.Cambio_scaler.transform(df1['Cambio'].values)
        df1['Blindado']      = self.Blindado_scaler.transform(df1['Blindado'].values)
        df1['Cor']           = self.Cor_scaler.transform(df1['Cor'].values)
        df1['UnicoDono']     = self.UnicoDono_scaler.transform(df1['UnicoDono'].values)
        df1['IPVAPago']      = self.IPVAPago_scaler.transform(df1['IPVAPago'].values)
        df1['Licenciado']    = self.Licenciado_scaler.transform(df1['Licenciado'].values)
        df1['TipoVendedor']  = self.TipoVendedor_scaler.transform(df1['TipoVendedor'].values)

        return df1

    def data_prediction(self, model, df2):
        y_pred = model.predict(df2)
        y_pred = np.expm1(y_pred).round(2)

        return y_pred
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from PIL import Image

header = st.container()
dataset1 = st.container()
dataset2 = st.container()
features1 = st.container()
features2 = st.container()

with header:
    st.title("Getaround analysis")


with dataset1:
    st.header("delays dataset")

    
    delays = pd.read_excel("data/get_around_delay_analysis.xlsx", engine="openpyxl")
    st.write(delays.head())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("rentals based on checkin type") 
        st.bar_chart(delays['checkin_type'].value_counts())
    with col2:
        st.header("rentals based on state")
        st.bar_chart(delays['state'].value_counts())
    
    with col3:
        no_negative_delays = delays[delays['delay_at_checkout_in_minutes'] > 0] #les retards négatifs ne m'intéresse pas je les supprime
        delay_without_outliers = no_negative_delays['delay_at_checkout_in_minutes'][no_negative_delays['delay_at_checkout_in_minutes'].between(no_negative_delays['delay_at_checkout_in_minutes'].quantile(.15), no_negative_delays['delay_at_checkout_in_minutes'].quantile(.85))]
        st.write(delays.isnull().sum()*100 / delays.shape[0])
        st.write("retard moyen aux checkout: ", int(np.round(delay_without_outliers.mean())), "minutes")
        st.write("ecart-type moyen pour les retards: ", int(np.round(delay_without_outliers.std())),"minutes")

    st.write("dans l'intervalle de 2 ecarts type à la moyenne, On a 95% de la population, je décide donc de fixer un délai minimum de la moyenne + 2 fois l'écart-type.")
    minimum_delay = int(np.round(delay_without_outliers.mean() + 2*delay_without_outliers.std()))
    st.write("délai minimum: ", minimum_delay, "minutes.")
    # je supprime également les outliers
    #print("retard moyen aux checkout: ", int(np.round(delay_without_outliers.mean())), "minutes")
    #print()
    #print("ecart-type moyen pour les retards: ", int(np.round(delay_without_outliers.std())),"minutes")
    # les locataires sont en moyenne en retard de 68 minutes
    # dans l'intervalle de 2 ecarts type à la moyenne, On a 95% de la population, je décide donc de fixer un délai minimum de la moyenne + 2 fois l'écart-type ce qui nous donne
    # Dans le but d'obtenir un délai minimum, concentrons nous sur les retards positifs et supprimons les valeurs abérrantes. En faisant la moyenne des retard, on observe que les locataires ont un retard moyen de 68 minutes, avec un ecart-type de 52 minutes, ce qui veut dire que les retars peuvent osciller entre 16 min et 2 heures.
    # Statistiquement, on sait que 95% des données se situent dans l'intervalle de 2 écarts type à la moyenne: je décide donc de fixer un délai minimum de la moyenne + 2 fois l'écart-type, ce qui nous donne un délai de 173 minutes.

    rentals_with_delta = delays[delays['time_delta_with_previous_rental_in_minutes'].isna() == False] #Ce qui nous intéresse pour l'instant ce sont les temps delta entre deux locations
    rentals_with_delta[rentals_with_delta['state'] == "canceled"]

    delays_for_scope = delays[delays['delay_at_checkout_in_minutes'].between(no_negative_delays['delay_at_checkout_in_minutes'].quantile(.15), no_negative_delays['delay_at_checkout_in_minutes'].quantile(.85))]

    delays_per_checkin_type = delays_for_scope.groupby('checkin_type')['delay_at_checkout_in_minutes'].mean() 
    # les moyennes des retards sont les mêmes que ce soit pour les checks connectés ou mobiles, on devrait donc appliquer ce délai minimum à toutes les voitures 

    st.bar_chart(delays_per_checkin_type)

with dataset2:
    st.header("pricing dataset")

    pricing = pd.read_csv("data/get_around_pricing_project.csv")
    st.write(pricing.head()) 

    col1, col2 = st.columns(2)
    with col1:
        daily_revenue = pricing['rental_price_per_day'].sum()
        #pricing['has_getaround_connect'].value_counts()

        mean_rental_by_model = pricing.groupby('model_key')['rental_price_per_day'].mean()
        df_mean_rental_by_model = pd.DataFrame(mean_rental_by_model.index)
        df_mean_rental_by_model['mean_rental'] = mean_rental_by_model.values
        
        mean_rental_price_per_day = pricing.groupby('has_getaround_connect')['rental_price_per_day'].mean()
        fig1, ax = plt.subplots()
        sns.barplot(x = mean_rental_price_per_day.index, y= mean_rental_price_per_day.values, )
        plt.title("Mean daily rental price with or without connect", loc='left', fontsize=20,bbox={'facecolor':'0.8', 'pad':8})
        sns.set_style('dark')
        st.pyplot(fig1)
        # On observe a peu près 20 euros

        mean_rental_price_per_car = int(np.round(pricing['rental_price_per_day'].mean()))

        ended_rentals = rentals_with_delta['previous_ended_rental_id'].count()
        generate_revenu = rentals_with_delta['previous_ended_rental_id'].count() * mean_rental_price_per_car
        impacted_rentals = rentals_with_delta[rentals_with_delta['time_delta_with_previous_rental_in_minutes'] < minimum_delay]['time_delta_with_previous_rental_in_minutes'].count()
        impacted_revenu = int(np.round((rentals_with_delta[rentals_with_delta['time_delta_with_previous_rental_in_minutes'] < minimum_delay]['time_delta_with_previous_rental_in_minutes'].count()) * 100 / rentals_with_delta['previous_ended_rental_id'].count()))

        rentals = [ended_rentals, impacted_rentals]

        fig4, ax = plt.subplots()
        ax.pie(rentals, labels=rentals, pctdistance=0.85, explode = (0.02, 0.02), colors=['bisque','darkred'])
        circle = plt.Circle( (0,0), 0.7, color='white')
        p = plt.gcf()
        p.gca().add_artist(circle)
        plt.legend(["all ended rentals", "impacted rentals"], loc='right')
        plt.title("impacted rentals on all rentals", fontsize=15,bbox={'facecolor':'0.8', 'pad':8})
        st.pyplot(fig4)


        #pricing.groupby('has_getaround_connect')['rental_price_per_day'].sum() #les revenus rapportés par les véhicules connectés ou non sont quasiment les mêmes



        only_for_delay = delays[delays['delay_at_checkout_in_minutes'].isna() == False]
        #only_for_delay[only_for_delay['delay_at_checkout_in_minutes'] > 0]['delay_at_checkout_in_minutes'].count() * 100 / only_for_delay.shape[0]
        # plus d'une personne sur deux est en retard pour le prochain check_in

        ended_rentals_with_delta = rentals_with_delta.shape[0]
        follow_canceled_rentals = rentals_with_delta[rentals_with_delta['state']=='canceled']['state'].count()
        percentage_canceled_rentals = int(np.round(rentals_with_delta[rentals_with_delta['state']=='canceled']['state'].count() * 100 / rentals_with_delta.shape[0]))

        cancel_after_rental = rentals_with_delta[rentals_with_delta['state']=='canceled']
        rentals_can_be_saved = int(np.round(cancel_after_rental[cancel_after_rental['time_delta_with_previous_rental_in_minutes'] < minimum_delay]['time_delta_with_previous_rental_in_minutes'].count() * 100 / rentals_with_delta[rentals_with_delta['state']=='canceled']['state'].count()))



    with col2:
        pie1 = go.Figure(
        go.Pie(
            labels = mean_rental_by_model.index,
            values = mean_rental_by_model.values,
            hoverinfo = "label+percent",
            textinfo = "value"
            ))
        st.plotly_chart(pie1)

        connect_cars = ["No", "Yes"]
        connect_cars_numbers = list((pricing['has_getaround_connect'].value_counts()).values)
        connect_cars_percentage = list((pricing['has_getaround_connect'].value_counts() * 100 / pricing.shape[0]).values)

        fig3, ax = plt.subplots()
        ax.pie(connect_cars_percentage, labels=connect_cars, autopct='%1.1f%%', pctdistance=0.85, explode = (0.02, 0.02), colors=['coral','lightgreen'])
        circle = plt.Circle( (0,0), 0.7, color='white')
        p = plt.gcf()
        p.gca().add_artist(circle)
        plt.legend(["non connected cars", "connected cars"], loc='right')
        plt.title("Percentage of cars based on getaround connect", fontsize=15,bbox={'facecolor':'0.8', 'pad':8})
        st.pyplot(fig3)















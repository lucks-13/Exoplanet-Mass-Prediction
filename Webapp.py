import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(
    page_title="NASA Exoplanet Mass Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_model_data():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model_data = load_model_data()

st.title("NASA Exoplanet Mass Prediction Dashboard")
st.caption("A Athena Award Project, Used all light weight high accuracy model derived from my analysis if you want highest accuracy use the below link for .pkl file")
st.caption("Uses Random Data as Source File")
st.markdown("[Click Here For Source Code](https://github.com/lucks-13/Exoplanet-Mass-Prediction)", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Model Selection")
model_cols = st.columns(4)
selected_models = []

for i, model_name in enumerate(list(model_data['trained_models'].keys())):
    col = model_cols[i % 4]
    with col:
        if st.checkbox(model_name, value=(i < 3)):
            selected_models.append(model_name)

st.markdown("---")

st.subheader("Input Parameters")

input_cols = st.columns(4)

with input_cols[0]:
    pl_rade = st.number_input("Planet Radius (Earth Radii)", value=5.0, min_value=0.1, max_value=100.0, step=0.1)
    pl_orbper = st.number_input("Orbital Period (days)", value=365.0, min_value=0.1, max_value=10000.0, step=1.0)
    pl_orbsmax = st.number_input("Orbital Distance (AU)", value=4.0, min_value=0.01, max_value=100.0, step=0.01)
    pl_eqt = st.number_input("Equilibrium Temperature (K)", value=288.0, min_value=0.0, max_value=3000.0, step=1.0)

with input_cols[1]:
    st_mass = st.number_input("Stellar Mass (Solar Masses)", value=7.0, min_value=0.1, max_value=10.0, step=0.1)
    st_rad = st.number_input("Stellar Radius (Solar Radii)", value=3.0, min_value=0.1, max_value=10.0, step=0.1)
    st_teff = st.number_input("Stellar Temperature (K)", value=5778.0, min_value=2000.0, max_value=10000.0, step=1.0)
    st_met = st.number_input("Stellar Metallicity", value=0.0, min_value=-2.0, max_value=1.0, step=0.01)

with input_cols[2]:
    st_age = st.number_input("Stellar Age (Gyr)", value=4.6, min_value=0.1, max_value=15.0, step=0.1)
    st_dens = st.number_input("Stellar Density (g/cm³)", value=1.4, min_value=0.1, max_value=10.0, step=0.1)
    pl_dens = st.number_input("Planet Density (g/cm³)", value=5.5, min_value=0.1, max_value=20.0, step=0.1)
    disc_year = st.number_input("Discovery Year", value=2020, min_value=1990, max_value=2025, step=1)

with input_cols[3]:
    sy_dist = st.number_input("System Distance (pc)", value=10.0, min_value=1.0, max_value=5000.0, step=1.0)
    discoverymethod = st.selectbox("Discovery Method", 
                                  options=['Transit', 'Radial Velocity', 'Imaging', 'Microlensing', 'Eclipse Timing Variations'])
    pl_masse_err = st.number_input("Planet Mass Uncertainty (%)", value=10.0, min_value=0.1, max_value=100.0, step=0.1)
    st_lum = st.number_input("Stellar Luminosity (Solar Luminosities)", value=2.0, min_value=0.01, max_value=1000.0, step=0.01)

def prepare_input(pl_rade, pl_orbper, pl_orbsmax, pl_eqt, st_mass, st_rad, st_teff, st_met, st_age, st_dens, pl_dens, discoverymethod, disc_year, sy_dist, pl_masse_err, st_lum):
    input_data = pd.DataFrame({
        'pl_rade': [pl_rade],
        'pl_orbper': [pl_orbper], 
        'pl_orbsmax': [pl_orbsmax],
        'pl_eqt': [pl_eqt],
        'st_mass': [st_mass],
        'st_rad': [st_rad],
        'st_teff': [st_teff],
        'st_met': [st_met],
        'st_age': [st_age],
        'st_dens': [st_dens],
        'pl_dens': [pl_dens],
        'discoverymethod': [discoverymethod],
        'disc_year': [disc_year],
        'sy_dist': [sy_dist],
        'pl_masse_err': [pl_masse_err],
        'st_lum': [st_lum]
    })
    
    estimated_mass = input_data['pl_rade'] ** 2.5 * input_data['pl_dens'] / 5.5
    input_data['mass_radius_ratio'] = estimated_mass / (input_data['pl_rade'] ** 3)
    input_data['stellar_planet_mass_ratio'] = input_data['st_mass'] / estimated_mass
    input_data['orbital_velocity'] = np.sqrt(input_data['st_mass'] / input_data['pl_orbsmax'])
    input_data['escape_velocity'] = np.sqrt(2 * estimated_mass / input_data['pl_rade'])
    input_data['hill_sphere'] = input_data['pl_orbsmax'] * (estimated_mass / (3 * input_data['st_mass'])) ** (1/3)
    
    age_bins = [0, 1, 5, 10, float('inf')]
    age_labels = [0, 1, 2, 3]
    input_data['stellar_age_category'] = pd.cut(input_data['st_age'], bins=age_bins, labels=age_labels, include_lowest=True).cat.codes
    
    input_data['planet_star_temp_ratio'] = input_data['pl_eqt'] / input_data['st_teff']
    
    dist_bins = [0, 100, 500, 1000, float('inf')]
    dist_labels = [0, 1, 2, 3]
    input_data['distance_category'] = pd.cut(input_data['sy_dist'], bins=dist_bins, labels=dist_labels, include_lowest=True).cat.codes
    
    input_data['log_orbital_period'] = np.log1p(input_data['pl_orbper'])
    input_data['sqrt_orbital_distance'] = np.sqrt(input_data['pl_orbsmax'])
    input_data['mass_temp_interaction'] = input_data['st_mass'] * input_data['pl_eqt']
    input_data['radius_period_interaction'] = input_data['pl_rade'] * input_data['pl_orbper']
    input_data['stellar_mass_metallicity'] = input_data['st_mass'] * input_data['st_met']
    
    poly_features = ['pl_rade', 'st_mass', 'pl_orbper']
    for feature in poly_features:
        input_data[f'{feature}_squared'] = input_data[feature] ** 2
        input_data[f'{feature}_cubed'] = input_data[feature] ** 3
    
    input_data['density_ratio'] = input_data['pl_dens'] / input_data['st_dens']
    input_data['log_planet_density'] = np.log1p(input_data['pl_dens'])
    
    if 'discoverymethod' in model_data['label_encoders']:
        le = model_data['label_encoders']['discoverymethod']
        try:
            input_data['discoverymethod'] = le.transform([discoverymethod])[0]
        except:
            input_data['discoverymethod'] = 0
    
    for col in model_data['feature_names']:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[model_data['feature_names']]
    input_data = input_data.replace([np.inf, -np.inf], 0)
    input_data = input_data.fillna(0)
    
    input_scaled = model_data['scaler'].transform(input_data)
    return input_scaled

if st.button("PREDICT EXOPLANET MASS", type="primary",use_container_width=True):
    if selected_models:
        input_scaled = prepare_input(pl_rade, pl_orbper, pl_orbsmax, pl_eqt, st_mass, st_rad, st_teff, st_met, st_age, st_dens, pl_dens, discoverymethod, disc_year, sy_dist, pl_masse_err, st_lum)
        
        predictions = {}
        for model_name in selected_models:
            model = model_data['trained_models'][model_name]
            pred = model.predict(input_scaled)[0]
            predictions[model_name] = max(0, pred)
        
        avg_prediction = np.mean(list(predictions.values()))
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        pred_cols = st.columns(3)
        
        with pred_cols[0]:
            st.metric("Average Prediction", f"{avg_prediction:.2f} Earth Masses")
        
        with pred_cols[1]:
            st.metric("Min Prediction", f"{min(predictions.values()):.2f} Earth Masses")
        
        with pred_cols[2]:
            st.metric("Max Prediction", f"{max(predictions.values()):.2f} Earth Masses")
        
        pred_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Prediction'])
        
        fig_pred = px.bar(pred_df, x='Model', y='Prediction', 
                         title='Predictions by Model',
                         color='Prediction',
                         color_continuous_scale='viridis')
        fig_pred.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_pred, use_container_width=True)
        
        st.dataframe(pred_df, use_container_width=True)
    else:
        st.error("Please select at least one model for prediction.")

st.markdown("---")

st.subheader("Model Performance Comparison")

scores_df = pd.DataFrame(model_data['model_scores']).T

chart_cols = st.columns(2)

with chart_cols[0]:
    fig_r2 = px.bar(x=scores_df.index, y=scores_df['r2'], 
                   title='R² Score by Model',
                   color=scores_df['r2'],
                   color_continuous_scale='blues')
    fig_r2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_r2, use_container_width=True)

with chart_cols[1]:
    fig_rmse = px.bar(x=scores_df.index, y=scores_df['rmse'], 
                     title='RMSE by Model',
                     color=scores_df['rmse'],
                     color_continuous_scale='reds')
    fig_rmse.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_rmse, use_container_width=True)

fig_scatter = px.scatter(scores_df, x='mae', y='r2', 
                        title='Model Performance: R² vs MAE',
                        hover_name=scores_df.index,
                        size='rmse')
st.plotly_chart(fig_scatter, use_container_width=True)


st.subheader("Feature Importance Analysis")

if model_data['feature_importance']:
    importance_data = []
    for model_name, importance in model_data['feature_importance'].items():
        for i, feat in enumerate(model_data['feature_names']):
            importance_data.append({
                'Model': model_name,
                'Feature': feat,
                'Importance': importance[i] if i < len(importance) else 0
            })
    
    importance_df = pd.DataFrame(importance_data)
    
    avg_importance = importance_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(15)
    
    fig_importance = px.bar(x=avg_importance.values, y=avg_importance.index, 
                           orientation='h',
                           title='Top 15 Features by Average Importance',
                           color=avg_importance.values,
                           color_continuous_scale='viridis')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    model_options = list(model_data['feature_importance'].keys())
    default_index = model_options.index("HuberRegressor")
    selected_model_importance = st.selectbox(
        "Select model for detailed feature importance:", options=model_options,index=default_index)

    
    if selected_model_importance:
        model_importance = model_data['feature_importance'][selected_model_importance]
        feat_imp_df = pd.DataFrame({
            'Feature': model_data['feature_names'][:len(model_importance)],
            'Importance': model_importance
        }).sort_values('Importance', ascending=False).head(20)
        
        fig_model_imp = px.bar(feat_imp_df, x='Importance', y='Feature', 
                              orientation='h',
                              title=f'Feature Importance - {selected_model_importance}',
                              color='Importance',
                              color_continuous_scale='plasma')
        st.plotly_chart(fig_model_imp, use_container_width=True)

st.subheader("Data Exploration")

viz_cols = st.columns(2)

with viz_cols[0]:
    fig_dist = px.histogram(model_data['df_clean'], x='pl_bmasse', 
                           title='Planet Mass Distribution',
                           nbins=50,
                           color_discrete_sequence=['skyblue'])
    st.plotly_chart(fig_dist, use_container_width=True)
    
    fig_corr = px.imshow(model_data['correlation_matrix'], 
                        title='Feature Correlation Matrix',
                        color_continuous_scale='RdBu_r',
                        aspect='auto')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    discovery_counts = model_data['df_clean']['discoverymethod'].value_counts()
    fig_discovery = px.pie(values=discovery_counts.values, names=discovery_counts.index,
                          title='Discovery Method Distribution')
    st.plotly_chart(fig_discovery, use_container_width=True)

with viz_cols[1]:
    df_mass = model_data['df_clean'][model_data['df_clean']['pl_bmasse'] > 0]
    
    fig_kde = px.density_contour(
    model_data['df_clean'],
    x='pl_bmasse',
    title='Planet Mass Density (KDE)',
    color_discrete_sequence=['lightcoral'])
    
    fig_kde.update_layout(
    xaxis_range=[-4000, 4000],
    yaxis_range=[-5000, 45000])
    
    st.plotly_chart(fig_kde, use_container_width=True)

    fig_scatter_mass_radius = px.scatter(model_data['df_clean'], x='pl_rade', y='pl_bmasse',
                                        title='Planet Mass vs Radius',
                                        opacity=0.6,
                                        color='pl_eqt',
                                        color_continuous_scale='viridis')
    st.plotly_chart(fig_scatter_mass_radius, use_container_width=True)
    
    yearly_discovery = model_data['df_clean']['disc_year'].value_counts().sort_index()
    fig_yearly = px.line(x=yearly_discovery.index, y=yearly_discovery.values,
                        title='Exoplanet Discoveries Over Time',
                        markers=True)
    fig_yearly.update_layout(xaxis_title='Discovery Year', yaxis_title='Number of Discoveries')
    st.plotly_chart(fig_yearly, use_container_width=True)

additional_viz_cols = st.columns(2)

with additional_viz_cols[0]:
    fig_stellar_mass = px.scatter(model_data['df_clean'], x='st_mass', y='pl_bmasse',
                                 title='Planet Mass vs Stellar Mass',
                                 opacity=0.6,
                                 color='st_teff',
                                 color_continuous_scale='plasma')
    st.plotly_chart(fig_stellar_mass, use_container_width=True)
    
    fig_orbital = px.scatter(model_data['df_clean'], x='pl_orbper', y='pl_bmasse',
                            title='Planet Mass vs Orbital Period',
                            log_x=True,
                            opacity=0.6,
                            color='pl_orbsmax',
                            color_continuous_scale='cividis')
    st.plotly_chart(fig_orbital, use_container_width=True)

with additional_viz_cols[1]:
    fig_temp = px.scatter(model_data['df_clean'], x='pl_eqt', y='pl_bmasse',
                         title='Planet Mass vs Equilibrium Temperature',
                         opacity=0.6,
                         color='sy_dist',
                         color_continuous_scale='turbo')
    st.plotly_chart(fig_temp, use_container_width=True)
    
    discovery_mass = model_data['df_clean'].groupby('discoverymethod')['pl_bmasse'].mean().sort_values(ascending=False)
    fig_discovery_mass = px.bar(x=discovery_mass.index, y=discovery_mass.values,
                               title='Average Planet Mass by Discovery Method',
                               color=discovery_mass.values,
                               color_continuous_scale='viridis')
    fig_discovery_mass.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_discovery_mass, use_container_width=True)

st.subheader("Advanced Visualizations")

adv_viz_cols = st.columns(2)

with adv_viz_cols[0]:
    fig_violin = px.violin(model_data['df_clean'], y='pl_bmasse', box=True,
                          title='Planet Mass Distribution (Violin Plot)')
    st.plotly_chart(fig_violin, use_container_width=True)
    
    fig_density = px.density_contour(model_data['df_clean'], x='pl_rade', y='pl_bmasse',
                                    title='Planet Mass vs Radius Density Plot')
    st.plotly_chart(fig_density, use_container_width=True)
    
    fig_box_method = px.box(model_data['df_clean'], x='discoverymethod', y='pl_bmasse',
                           title='Planet Mass by Discovery Method (Box Plot)')
    fig_box_method.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_box_method, use_container_width=True)

with adv_viz_cols[1]:
    fig_3d = px.scatter_3d(model_data['df_clean'].sample(500), x='pl_rade', y='st_mass', z='pl_bmasse',
                          color='pl_eqt', title='3D Planet-Star Relationship',
                          color_continuous_scale='viridis')
    st.plotly_chart(fig_3d, use_container_width=True)
    
    fig_sunburst = px.sunburst(model_data['df_clean'], path=['discoverymethod'], values='pl_bmasse',
                              title='Planet Mass by Discovery Method (Sunburst)')
    st.plotly_chart(fig_sunburst, use_container_width=True)
    
    stellar_temp_bins = pd.cut(model_data['df_clean']['st_teff'], bins=5)
    temp_mass_data = model_data['df_clean'].groupby(stellar_temp_bins)['pl_bmasse'].agg(['mean', 'count']).reset_index()
    temp_mass_data['st_teff_str'] = temp_mass_data['st_teff'].astype(str)
    fig_polar = px.bar_polar(temp_mass_data, r='mean', theta='st_teff_str',
                            title='Average Planet Mass by Stellar Temperature (Polar Plot)')
    st.plotly_chart(fig_polar, use_container_width=True)

st.subheader("Statistical Analysis")

stat_cols = st.columns(2)

with stat_cols[0]:
    mass_temp_corr = model_data['df_clean'][['pl_bmasse', 'pl_eqt', 'st_teff', 'pl_rade']].corr()
    fig_corr_focused = px.imshow(mass_temp_corr, title='Key Parameters Correlation',
                                color_continuous_scale='RdBu_r', text_auto=True)
    st.plotly_chart(fig_corr_focused, use_container_width=True)

    fig_box = px.box(
    model_data['df_clean'],
    x='pl_bmasse',
    title='Planet Mass Distribution (Box Plot)',
    color_discrete_sequence=['lightcoral'])
    st.plotly_chart(fig_box, use_container_width=True)

with stat_cols[1]:
    disc_year_mass = model_data['df_clean'].groupby('disc_year')['pl_bmasse'].agg(['mean', 'std']).reset_index()
    fig_trend = px.scatter(disc_year_mass, x='disc_year', y='mean', error_y='std',
                          title='Planet Mass Trend Over Discovery Years',
                          trendline='ols')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    fig_hexbin = px.density_heatmap(model_data['df_clean'], x='st_mass', y='pl_bmasse',
                                   title='Planet Mass vs Stellar Mass (Density Heatmap)')
    st.plotly_chart(fig_hexbin, use_container_width=True)

st.subheader("Interactive Analysis")

interact_cols = st.columns(2)

with interact_cols[0]:
    sample_data = model_data['df_clean'].sample(min(1000, len(model_data['df_clean'])))
    fig_animated = px.scatter(sample_data, x='pl_rade', y='pl_bmasse',
                             animation_frame='disc_year', color='discoverymethod',
                             title='Planet Discoveries Animation by Year',
                             range_x=[0, sample_data['pl_rade'].quantile(0.95)],
                             range_y=[0, sample_data['pl_bmasse'].quantile(0.95)])
    st.plotly_chart(fig_animated, use_container_width=True)

with interact_cols[1]:
    fig_treemap = px.treemap(model_data['df_clean'], path=['discoverymethod'], values='pl_bmasse',
                            title='Planet Mass Distribution by Discovery Method (Treemap)')
    st.plotly_chart(fig_treemap, use_container_width=True)

st.markdown("---")
st.subheader("Dataset Statistics")

stats_cols = st.columns(4)

with stats_cols[0]:
    st.metric("Total Exoplanets", len(model_data['df_clean']))

with stats_cols[1]:
    st.metric("Features Used", len(model_data['feature_names']))

with stats_cols[2]:
    st.metric("Models Trained", len(model_data['trained_models']))

with stats_cols[3]:
    best_model = max(model_data['model_scores'].items(), key=lambda x: x[1]['r2'])
    st.metric("Best Model", f"{best_model[0]}")

summary_stats = model_data['df_clean'][['pl_bmasse', 'pl_rade', 'st_mass', 'pl_orbper']].describe()
st.dataframe(summary_stats, use_container_width=True)

st.markdown("---")
st.markdown("*Data source: NASA Exoplanet Archive*")

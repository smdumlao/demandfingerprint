"""
The ``solgen`` module is a collection of functions used to calculate the
power generation of solar panels using the pvlib library.
"""

import numpy as np
import pandas as pd

import pvlib

fukuoka_loc = pvlib.location.Location(latitude = 33.5833,
                                      longitude = 130.3833,
                                      tz = 'Asia/Tokyo',
                                      altitude = 2.5,
                                      name = 'Fukuoka')

def date_range_localized(df, tz_area = 'Asia/Tokyo'):
    '''
    Reconstructs the index of the df with the tz
    '''
    df_index = df.index
    dfx = df.copy()
    dfx.index = pd.date_range(start=df_index[0], end=df_index[-1],
                              freq='H', tz=tz_area)
    return dfx

def build_weather_data(ghi_df, temp_df, loc, wspd_df=None):
    '''
    Builds the weather data for the weather_to_power calculation.
    '''

    #align index
    index_start = ghi_df.index[0]
    index_end = ghi_df.index[-1]

    temp_df = temp_df[index_start:index_end].to_frame('DryBulb')

    # concat df
    weather_data = pd.concat([ghi_df, temp_df], axis=1)

    # localize and adjust the datetimeindex
    weather_data = date_range_localized(weather_data)
    weather_data = weather_data.shift(freq='-30Min')


    # calculate solpos
    solpos = pvlib.solarposition.get_solarposition(weather_data.index,
                                                   loc.latitude, loc.longitude)

    # calculate the dni dhi
    erbs = pvlib.irradiance.erbs(weather_data['ghi'], solpos['zenith'],
                             weather_data.index, min_cos_zenith=0.065, max_zenith=85)

    weather_data['dni'] = erbs['dni']
    weather_data['dhi'] = erbs['dhi']

    if wspd_df == None:
        weather_data['Wspd'] = 1 #default windspeed

    weather_data = weather_data.shift(freq='30Min')

    return weather_data

def weather_to_power(weather_data, loc, panel_config):
    '''
    Adapted from tmy_to_power tutorial of pvlib library
    '''
    # TMY data seems to be given as hourly data with time stamp at the end
    # shift the index 30 Minutes back for calculation of sun positions
    weather_data = weather_data.shift(freq='-30Min')

    # solar position
    solpos = pvlib.solarposition.get_solarposition(weather_data.index,
                                                   loc.latitude, loc.longitude)

    # dni extra: extra terrestrial radiation
    dni_extra = pvlib.irradiance.get_extra_radiation(weather_data.index)
    dni_extra = pd.Series(dni_extra, index=weather_data.index)

    # airmass
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])

    # poa sky diffuse
    poa_sky_diffuse = pvlib.irradiance.haydavies(panel_config['surface_tilt'],
                                                 panel_config['surface_azimuth'],
                                                 weather_data['dhi'], weather_data['dni'],
                                                 dni_extra, solpos['apparent_zenith'],
                                                 solpos['azimuth'])

    # poa ground diffuse
    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(panel_config['surface_tilt'],
                                                             weather_data['ghi'],
                                                             panel_config['albedo'])

    # angle of incidence
    aoi = pvlib.irradiance.aoi(panel_config['surface_tilt'], panel_config['surface_azimuth'],
                               solpos['apparent_zenith'], solpos['azimuth'])

    # total plane of array (POA) irradiance
    poa_irrad = pvlib.irradiance.poa_components(aoi, weather_data['dni'], poa_sky_diffuse,
                                                poa_ground_diffuse)

    # cell temperature
    thermal_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
    pvtemps = pvlib.temperature.sapm_cell(poa_irrad['poa_global'], weather_data['DryBulb'],
                                          weather_data['Wspd'], **thermal_params)

    # DC power using SAPM
    pv_modules =pvlib.pvsystem.retrieve_sam(name='SandiaMod')
    sandia_modules = pvlib.pvsystem.retrieve_sam(name='SandiaMod')
    sandia_module = sandia_modules[panel_config['panel']]

    # effective irradiance
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_irrad.poa_direct,
                                                                    poa_irrad.poa_diffuse,
                                                                    airmass, aoi, sandia_module)
    sapm_out = pvlib.pvsystem.sapm(effective_irradiance, pvtemps, sandia_module)

    # DC power using single diode
    cec_modules = pvlib.pvsystem.retrieve_sam(name='CECMod')
    cec_module = cec_modules.Canadian_Solar_Inc__CS5P_220M

    d = {k: cec_module[k] for k in ['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s']}

    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = (
        pvlib.pvsystem.calcparams_desoto(poa_irrad.poa_global,
                                         pvtemps,
                                         cec_module['alpha_sc'],
                                         EgRef=1.121,
                                         dEgdT=-0.0002677, **d))

    # single_diode_out = pvlib.pvsystem.singlediode(photocurrent,
    #                                               saturation_current,
    #                                               resistance_series,
    #                                               resistance_shunt,
    #                                               nNsVth)

    # AC power using SAPM
    sapm_inverters = pvlib.pvsystem.retrieve_sam('sandiainverter')
    sapm_inverter = sapm_inverters[panel_config['inverter']]

    p_acs = pd.DataFrame()
    p_acs['sapm'] = pvlib.inverter.sandia(sapm_out.v_mp, sapm_out.p_mp, sapm_inverter)
    # p_acs['sd'] = pvlib.inverter.sandia(single_diode_out.v_mp, 
    #                                     single_diode_out.p_mp,
    #                                     sapm_inverter)

    #clean the dataframe
    p_acs = p_acs.fillna(0)
    p_acs = p_acs.where(p_acs > 0, 0)

    #move forward
    p_acs = p_acs.shift(freq='30Min')
    return p_acs

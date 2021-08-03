from analyse_data_utils.estimate_number_occurrences \
    import estimate_number_occurrences

if __name__ == "__main__":
    # like this it can be run also when importing this in
    # generate_original_concatenated_data

    import yaml
    from yaml import Loader

    with open("../configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    # The 5 epochs to analyze
    epochs = config['epochs']
    # The 4 trial types to analyze
    trialtypes = config['trialtypes']
    # The sessions to analyze
    sessions = config['sessions']
    # Absolute minimum number of occurrences of a pattern
    abs_min_occ = config['abs_min_occ']
    # Magnitude of the binsize used
    binsize = config['binsize']
    # The percentile for the Poisson distribution to fix minimum number of occ
    percentile_poiss = config['percentile_poiss']
    # The percentile for the Poisson distribution of rates
    percentile_rates = config['percentile_rates']
    # minimum number of spikes per patterns
    abs_min_spikes = config['abs_min_spikes']
    # The winlen parameter for the SPADE analysis
    winlen = config['winlen']
    # Spectrum to use
    spectrum = config['spectrum']
    # Dithering to use to generate surrogates in seconds
    dither = config['dither']
    # Number of surrogates to generate
    n_surr = config['n_surr']
    # Significance level
    alpha = config['alpha']
    # Multitesting statistical correction
    correction = config['correction']
    # PSR parameters
    psr_param = config['psr_param']
    # Unit in which every time of the analysis is expressed
    unit = config['unit']
    # Firing rate threshold to possibly exclude neurons
    firing_rate_threshold = config['firing_rate_threshold']
    # Surrogate method to use
    surr_method = config['surr_method']

    # loading parameters
    param_dict, excluded_neurons = \
        estimate_number_occurrences(
            sessions=sessions,
            epochs=epochs,
            trialtypes=trialtypes,
            binsize=binsize,
            abs_min_spikes=abs_min_spikes,
            abs_min_occ=abs_min_occ,
            correction=correction,
            psr_param=psr_param,
            alpha=alpha,
            n_surr=n_surr,
            dither=dither,
            spectrum=spectrum,
            winlen=winlen,
            percentile_poiss=percentile_poiss,
            percentile_rates=percentile_rates,
            unit=unit,
            firing_rate_threshold=firing_rate_threshold,
            surr_method=surr_method)

% Author-exact SWIPE' reference, for cross-checking the golden.
%
% Runs Camacho's OWN dissertation-appendix MATLAB (swipep + helpers, copied
% verbatim from kylebgorman/swipe's swipe.m) under Octave on the committed
% fixtures, and prints the median voiced f0 per fixture. Compare these to
% make_swipe_golden.py's numpy port (should agree to ~0.1 Hz) and to the
% Rust engine::pitch::swipe.
%
% Prereqs:  base Octave only (no signal package) — `hanning` and `specgram`
%           are defined locally below so the swipep + kernel code (the part
%           that matters) stays Camacho-verbatim and runs anywhere.
% Run:      octave --no-gui crates/engine/tests/dsp/swipe/run_swipe_octave.m
%
% Only deviation from Camacho's file: wavread -> reading our .tsv fixtures.

1;

% Local stand-ins for the signal-package functions swipep relies on, so no
% package install is needed. hanning is Camacho's (denominator N+1); specgram
% is the one-sided STFT swipep expects (f spans 0..fs/2 to cover fERBs).
function w = hanning(n)
    w = 0.5 * (1 - cos(2*pi*(1:n)'/(n+1)));
end

function [S, f, t] = specgram(x, nfft, fs, window, noverlap)
    step = nfft - noverlap;
    ncol = 1 + floor((length(x) - nfft)/step);
    S = zeros(nfft/2+1, ncol);
    for m = 1:ncol
        seg = x((m-1)*step + (1:nfft)) .* window;
        X = fft(seg);
        S(:,m) = X(1:nfft/2+1);
    end
    f = (0:nfft/2)' * fs / nfft;
    t = ((0:ncol-1)*step + nfft/2)' / fs;
end

function [p,t,s] = swipep(x,fs,plim,dt,sTHR)
    if ~ exist( 'plim', 'var' ) || isempty(plim), plim = [30 5000]; end
    if ~ exist( 'dt', 'var' ) || isempty(dt), dt = 0.01; end
    dlog2p = 1/96;
    dERBs = 0.1;
    if ~ exist( 'sTHR', 'var' ) || isempty(sTHR), sTHR = -Inf; end
    t = [ 0: dt: length(x)/fs ]';
    dc = 4; K = 2;
    log2pc = [ log2(plim(1)): dlog2p: log2(plim(end)) ]';
    pc = 2 .^ log2pc;
    S = zeros( length(pc), length(t) );
    logWs = round( log2( 4*K * fs ./ plim ) );
    ws = 2.^[ logWs(1): -1: logWs(2) ];
    pO = 4*K * fs ./ ws;
    d = 1 + log2pc - log2( 4*K*fs./ws(1) );
    fERBs = erbs2hz([ hz2erbs(pc(1)/4): dERBs: hz2erbs(fs/2) ]');
    for i = 1 : length(ws)
        dn = round( dc * fs / pO(i) );
        xzp = [ zeros( ws(i)/2, 1 ); x(:); zeros( dn + ws(i)/2, 1 ) ];
        w = hanning( ws(i) );
        o = max( 0, round( ws(i) - dn ) );
        [ X, f, ti ] = specgram( xzp, ws(i), fs, w, o );
        M = max( 0, interp1( f, abs(X), fERBs, 'spline', 0) );
        L = sqrt( M );
        if i==length(ws)
            j = find(d - i > -1);
            k = find(d(j) - i < 0);
        elseif i==1
            j = find(d - i < 1);
            k = find(d(j) - i > 0);
        else
            j = find(abs(d - i) < 1);
            k = (1:length(j))';
        end
        Si = pitchStrengthAllCandidates( fERBs, L, pc(j) );
        if size(Si,2) > 1
            Si = interp1( ti, Si', t, 'linear', NaN )';
        else
            Si = repmat( NaN, length(Si), length(t) );
        end
        lambda = d( j(k) ) - i;
        mu = ones( size(j) );
        mu(k) = 1 - abs( lambda );
        S(j,:) = S(j,:) + repmat(mu,1,size(Si,2)) .* Si;
    end
    p = repmat( NaN, size(S,2), 1 );
    s = repmat( NaN, size(S,2), 1 );
    for j = 1 : size(S,2)
        [ s(j), i ] = max( S(:,j) );
        if s(j) < sTHR, continue, end
        if i==1
             p(j)=pc(1);
        elseif i==length(pc)
            p(j)=pc(1);
        else
            I = i-1 : i+1;
            tc = 1 ./ pc(I);
            ntc = ( tc/tc(2) - 1 ) * 2*pi;
            c = polyfit( ntc, S(I,j), 2 );
            ftc = 1 ./ 2.^[ log2(pc(I(1))): 1/12/64: log2(pc(I(3))) ];
            nftc = ( ftc/tc(2) - 1 ) * 2*pi;
            [s(j) k] = max( polyval( c, nftc ) );
            p(j) = 2 ^ ( log2(pc(I(1))) + (k-1)/12/64 );
        end
    end
    p(isnan(s)) = NaN;
end

function S = pitchStrengthAllCandidates( f, L, pc )
    L = L ./ repmat( sqrt( sum(L.*L) ), size(L,1), 1 );
    S = zeros( length(pc), size(L,2) );
    for j = 1 : length(pc)
        S(j,:) = pitchStrengthOneCandidate( f, L, pc(j) );
    end
end

function S = pitchStrengthOneCandidate( f, L, pc )
    n = fix( f(end)/pc - 0.75 );
    k = zeros( size(f) );
    q = f / pc;
    for i = [ 1 primes(n) ]
        a = abs( q - i );
        p = a < .25;
        k(p) = cos( 2*pi * q(p) );
        v = .25 < a & a < .75;
        k(v) = k(v) + cos( 2*pi * q(v) ) / 2;
    end
    k = k .* sqrt( 1./f );
    k = k / norm( k(k>0) );
    S = k' * L;
end

function erbs = hz2erbs(hz), erbs = 21.4 * log10( 1 + hz/229 ); end
function hz = erbs2hz(erbs), hz = ( 10 .^ (erbs./21.4) - 1 ) * 229; end

% ---- runner over the committed fixtures ----
here = fileparts( mfilename('fullpath') );
cases = { 'f0_150', 150; 'f0_220', 220; 'f0_330', 330 };
fs = 16000; plim = [100 600]; dt = 0.01; sTHR = 0.30;
fid = fopen( fullfile(here, 'swipe_golden.tsv'), 'w' );
fprintf( fid, 'true_f0\tswipe_median_hz\tfs\t%d\tdt\t%g\tsthr\t%g\tplim_lo\t%g\tplim_hi\t%g\n', ...
         fs, dt, sTHR, plim(1), plim(2) );
printf('fixture   true   octave-median\n');
for ci = 1:size(cases,1)
    x = dlmread( fullfile(here, [cases{ci,1} '_input.tsv']), '\t', 1, 0 );
    [p,t,s] = swipep( x(:,1), fs, plim, dt, sTHR );
    med = median( p(~isnan(p)) );
    printf('%-9s %5d   %8.2f Hz\n', cases{ci,1}, cases{ci,2}, med);
    fprintf( fid, '%.8e\t%.8e\n', cases{ci,2}, med );
end
fclose(fid);

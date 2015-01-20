#------------------------------------------------------------------------------------------
# Purpose:          Python-coded statistical toolbox for methods described in the Reference
# Reference:        Statistical Methods in Water Resources (SMIWR) by Helsel, Hirsch, 2002
# Code Assimilator: KSedmera
# Last Update:      1/2015
#------------------------------------------------------------------------------------------
from __future__ import print_function
import os, sys, time, csv, fileinput, Tkinter as Tk, tkFileDialog
try:    # Supplemental Python libraries required by the toolbox
    import numpy as np, pylab, scipy, scipy.stats as stats, pandas, datetime
    from matplotlib.ticker import AutoMinorLocator
    from dateutil.parser import parse as parsedate
    from textwrap import wrap
    #import AnnoteFinder as AF
except ImportError as exc:
    import subprocess
    sys.stderr.write("Error: {}. Closing in 5 sec...\n".format(exc))
    print("Note: These tools require Python 2.6.5 - 2.7.5 (e.g. 2.6.5 comes with ArcGIS 10.0),")
    print("      AND several free science-related Python libraries. See which one your missing above.")
    time.sleep(5);  sys.exit()

ls = os.linesep
fs = os.pathsep

def FuncSelectRun(funclist):
    """ Usage: FuncSelectNRun(funclist)
    Populates a Tk Listbox with this library's 'funclist' (list).
    It lets the user select an function, 'funcq', and function argument, 'argq'
    (method), e.g. funcq(argq) you want the ListBox to run.
    """
    print("Use the new Tk window to select an Function to execute.\nClose all Tk windows to exit this routine.")
    print('\nYour variable names include:\n{}'.format(', '.join(cnms)))
    master = Tk.Tk()
    master.title('Function Selector')
    F1 = Tk.Frame(master)
    lab = Tk.Label(F1)
    lab.config(text="Select an function, then press a button below.")
    lab.pack()
    s = Tk.Scrollbar(F1)
    L = Tk.Listbox(F1, width=75)
    s.pack(side=Tk.RIGHT, fill=Tk.Y)
    L.pack(side=Tk.LEFT, fill=Tk.Y)
    s['command'] = L.yview
    L['yscrollcommand'] = s.set
    for id in funclist:    L.insert(Tk.END, id)
    F1.pack(side=Tk.TOP)
    L.selection_set(0); L.activate(0)

    F2 = Tk.Frame(master)
    ec = Tk.Entry(F2, width=75)
    ec.insert(0, "[SPECIFY ARGS]")
    ec.pack(side=Tk.LEFT)
    F2.pack(side=Tk.TOP)

    def DisplayDocstring():
        print("\n{0}__doc__:".format(L.get(Tk.ACTIVE)))
        pds = "print({0}.__doc__)".format(L.get(Tk.ACTIVE))
        eval(pds)

    def RunFunc():
        funcq = L.get(Tk.ACTIVE)
        argq = ec.get()
        if argq == "[SPECIFY ARGS]" or argq == "":
            print("No function argments were specified. Try again.")
        else:
            runstr = """{0}({1})""".format(funcq, argq)
            eval(runstr)
            print('\nYour "data" variables include:\n  {0}'.format(', '.join(cnms)))
            if dataV.keys():    print('Your "dataV" variables include:\n  {0}'.format(', '.join(dataV.keys())))

    F4 = Tk.Frame(master)
    b1 = Tk.Button(F4, text="Print Docstring", command=DisplayDocstring)
    b1.pack(side=Tk.LEFT)
    b2 = Tk.Button(F4, text="Run", command=RunFunc)
    b2.pack(side=Tk.LEFT)
    F4.pack()
    Tk.mainloop()

def SelectFile(req = 'Please select a {} file:', ft='csv'):
    """ Customizable file-selection dialogue window, returns list() = [full path, root path, and filename]. """
    try:    # Try to select a file
        foptions = dict(filetypes=[(ft+' file','*.'+ft)], defaultextension='.'+ft)
        root = Tk.Tk(); root.withdraw(); root.attributes("-topmost", True); fname = tkFileDialog.askopenfilename(title=req.format(ft), **foptions); root.destroy()
        return [fname] + list(os.path.split(fname))
    except: print("Error: {0}".format(sys.exc_info()[1])); time.sleep(5);  sys.exit()

def Col2list(c,fil=[]):
    """ Usage: Col2list(c,fil)
    c   => column name of the data to export, in quotes, e.g. "Variable1"
    fil => a value-filtering list, e.g. ["Variable2", "> 2.5"]
    Returns a list() of data from column c"""
    if fil: exec 'out = data[data[fil[0]]'+fil[1]+'][c].values.tolist()'; return out
    else:   return data[:][c].values.tolist()

def outfile(fnmt,lbl,ext='csv'):
    """outfile(fnmt, lbl, ext = 'csv')
    Generates an output filename like "fnm1" with custom suffixes "lbl" and "ext"."""
    if "_" in fnmt:  fnmop = fnmt.split('_')[0]
    elif "-" in fnmt:  fnmop = fnmt.split('-')[0]
    elif len(fnmt.split('.')) < 4:  fnmop = fnmt.split('.')[0]
    else:   fnmop = fnmt.split('.')[0][:4]
    return os.path.join(fpath, '{0}_{1}_{2}.{3}'.format(fnmop, lbl, time.strftime('%Y%m%d-%H%M'), ext))

def SaveOutputCSV(dataL, vnms, lbl):
    """SaveOutputCSV(dataL,vnms,lbl)
    dataL:  [list1, list2,...], a list of lists, where each data list must be
              the same length.
    vnms:   ['Col1Name', 'Col2name',...], a list of strings with the desired
              header for each variable.
    lbl:    'FileNamelabel', a string with the desired filename label."""
    fnmo = outfile(fnm1, lbl)
    with open(fnmo, 'ab') as fo:
        nv, nl = len(dataL), len(dataL[0])
        fo.write(','.join(vnms))
        for line in range(nl):
            fo.write(ls+','.join(str(dataL[n][line]) for n in range(nv)))
    print('- Wrote {0} records to {1}.'.format(nl+1, fnmo.split(fs)[-1]))

def UniqueIDs2List(c):
    """UniqueIDs2List(c) Returns a tuple of the unique IDs in column, c """
    return tuple(set(data[:][cnms[c]].values.tolist()))

def ListBoxSelectNRun(idlist, PlotMethod, tit, ExtraCommand='None'):
    """ListBoxSelectNRun(idlist, PlotMethod, tit, ExtraCommand='None')
    Usage: ListBoxSelectNRun(idlist, PlotMethod, tit, ExtraCommand)
    Populates a Tk Listbox with an 'idlist' (list) and 'tit' (string).
    It lets the user select an ID, 'idq', to pass into whatever 'PlotMethod'
    (method), e.g. PlotMethod(idq) you want the ListBox to run.
    * PlotMethod must be a method that knows where to find the data and idlist
    to be used, for example to plot points / lines in a new figure.
    * ExtraCommand='pylab.show()', which is optional, for example, allows you to
    follow a plotting method with a figure-closing command, when you want it to
    only plot a single line in a figure (rather than multiple lines in a single
    figure by using PlotMethod without this ExtraCommand). """
    print("Use the new Tk window to select an ID to "+tit+".\nClose all Tk windows to exit this routine.")
    master = Tk.Tk()
    master.title(tit+' ID Selector')
    F1 = Tk.Frame(master)
    lab = Tk.Label(F1)
    lab.config(text="Select an ID, then press a button below.")
    lab.pack()
    s = Tk.Scrollbar(F1)
    L = Tk.Listbox(F1, width=75)
    s.pack(side=Tk.RIGHT, fill=Tk.Y)
    L.pack(side=Tk.LEFT, fill=Tk.Y)
    s['command'] = L.yview
    L['yscrollcommand'] = s.set
    for id in idlist:    L.insert(Tk.END, id)
    F1.pack(side=Tk.TOP)
    L.selection_set(0); L.activate(0)

    def PlotSID():
        idq = L.get(Tk.ACTIVE)
        PlotMethod(idq)
        eval(ExtraCommand)
    def PlotNID():
        if L.get(Tk.ACTIVE) == idlist[-1]:  print('End of list. Select a previous ID, or close to quit.')
        else:
            nid = idlist.index(L.get(Tk.ACTIVE))+1
            idq = idlist[nid]
            L.selection_clear(nid-1); L.selection_set(nid); L.activate(nid)
            PlotMethod(idq)
            eval(ExtraCommand)

    F4 = Tk.Frame(master)
    l1 = Tk.Label(F4, text=tit+": ")
    l1.pack(side=Tk.LEFT)
    b1 = Tk.Button(F4, text="Selected ID", command=PlotSID)
    b1.pack(side=Tk.LEFT)
    b2 = Tk.Button(F4, text=" Next ID", command=PlotNID)
    b2.pack(side=Tk.LEFT)
    F4.pack()
    Tk.mainloop()

def LoadTB(TBn):
    """LoadTB(TBn)
    This function assumes the csv tables are in the same directory as this script. """
    if TBn == 4:    fn = os.path.normpath(sys.path[0]+'/'+'RankSumTableB4.csv')
    elif TBn == 6:  fn = os.path.normpath(sys.path[0]+'/'+'RankSumTableB6.csv')
    else:   print("Error: Unknown table."); sys.exit()
    global TB
    try:    TB = pandas.read_csv(fn)
    except: print("Error:", sys.exc_info()[1]); return None

def Lookupx(TBn,f1,f2,x,xt):
    """Lookupx(TBn,f1,f2,x,xt)
        http://stackoverflow.com/questions/8916302/selecting-across-multiple-columns-with-python-pandas
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html """
    try:
        if TBn == 4:
            t ='TB['+' & '.join(["(TB['%s']==%s)" % (i,j) for i,j in [('n',f1),('m',f2)]])+']'
            exec 'ca='+t; exec "cx=np.array(ca[['"+xt+"']].values).T[0]"; exec "cp=np.array(ca[['p']].values).T[0]"
            cxp= np.column_stack((cx,cp));  cxps = cxp[np.lexsort((cxp[:,0],cxp[:,0]))]
            spl = scipy.interpolate.UnivariateSpline(cxps[:,0].astype(int), cxps[:,1], s=1)
            if x >= min(cxps[:,0]) and x <= max(cxps[:,0]): px = round(spl(x),3)
            else:   px = round(spl(x),4)
            return px
        elif TBn == 6:
            alphas = [float(i.strip('a')) for i in TB.columns.tolist()[2:6]]
            t ='TB['+' & '.join(["(TB['%s']==%s)" % (i,j) for i,j in [('rej',f1),('n',f2)]])+']'
            exec 'cws=np.array('+t+')[0][2:6]'
            cwa= np.column_stack((cws,alphas)); cwas = cwa[np.lexsort((cwa[:,0],cwa[:,0]))]
            spl = scipy.interpolate.UnivariateSpline(cwas[:,0].astype(int), cwas[:,1], s=1)
            if x >= min(cwas[:,0]) and x <= max(cwas[:,0]): px = round(spl(x),3)
            elif spl(x) < 0:    px = 0
            else:   px = round(spl(x),4)
            return px
    except: print('Error:', sys.exc_info()[1]); return None

def AddDateCol(fullDateCol='Date', dtcoltype='year'):
    """SM_AddDateCol(data, fullDateCol='Date', dtcoltype='year')
    Appends a variable/column to "data" with a name like "Date-year", that has
    the desired date/time-part based on the "fullDateCol" column-name and
    "dtcoltype" date/time-part that you specify.
    The dateutil.parser.parse() method can usually extract the following
    date/time-parts that might be in your "fullDateCol" field:
    - year, month, day, weekday, hour, minute, second."""
    cnm = '-'.join([fullDateCol,dtcoltype])
    exec "data['{0}'] = [i.{2} for i in data['{1}']]".format(cnm,fullDateCol,dtcoltype)
    cnms.append(cnm)
    print('\nSuccessfully added "{}" to the "data" variables.'.format(cnm))
    return data, cnms

def AddDiffCol(cnm,c1,c2,operator='-'):
    """SM_AddDiffCol(cnm,c1,c2,operator='-')
    Appends a column named <cnm> to the DataFrame, "data", where the "operator"
    is used to '+', '-', '*', '/' the data in columns c1 and c2
    (i.e. c1 <operator> c2)."""
    exec "data[cnm] = data[:][c1] {} data[:][c2]".format(operator)
    cnms.append(cnm)
    print('\nSuccessfully added "{}" to the "data" variables.'.format(cnm))
    #ft = fname.split('.'); fnout = ft[0]+'_'+cnm+'.'+ft[1]
    #data.to_csv(fnout, cols=(cnms[c1],cnms[c2],cnm), index=False)
    return data, cnms

def AddCalculatedVar(Col2bMasked,ColAggregator='',aggregator='np.max'):
    """AddIntegrationVar(Col2bMasked,ColAggregator='',aggregator='np.max')
    The "aggregator" method can be any function in the Python/Numpy
      builtins (e.g. np.max, np.min, np.sum, np.mean, np.exp, np.sqrt, np.log).
      However, the latter three example operators should only used with
      ColAggregator=''. Note, use "np.<operator>" for Numpy operators.
    Adds a new variable with one of two name-patterns.
    If ColAggregator='', then the new variable (e.g. named <Col2bMasked>_Exp)
      will only be assiged to a new list, where the "aggregator" is only
      applied to the individual <Col2bmasked> values. The output list will be
      identical in length to the non-aggregated <Col2bMasked> column.
    If ColAggregator='<AValidColName>', then the new variables will be added to
      the "dataV" variable-dictionary. The first will have name like
      <Col2bMasked>_<ColAggregator>, and the second will have a name like
      <Col2bMasked>_MaxOf<ColAggregator>. This method assumes that the
      <ColAggregator> data-column has repeated values, which will be used to
      aggregate the associated <Col2bMasked> values. The new variable will thus
      only contain values corresponding to the unique values in the
      <ColAggregator> data column (e.g. the year of a set of daily/monthly
      measurements)."""
    # Refs:
    # http://bconnelly.net/2013/10/summarizing-data-in-python-with-pandas/
    if ColAggregator=='':
        vnm = '_'.join([Col2bMasked, aggregator.split('.')[-1].title()])
        exec "dataV['{0}'] = data['{1}'].apply({2}).values.tolist()".format(vnm, Col2bMasked, aggregator)
    else:
        vnm1 = '_'.join([Col2bMasked, ColAggregator])
        vnm2 = '_'.join([Col2bMasked, aggregator.split('.')[-1].title()+'Of'+ColAggregator])
        byColAggregator = data.groupby(ColAggregator)
        exec "dataV['{0}'] = dict(list(byColAggregator)).keys()".format(vnm1)
        exec "dataV['{0}'] = byColAggregator['{1}'].aggregate({2}).values.tolist()".format(vnm2, Col2bMasked, aggregator)
    print('\nSuccessfully added the "{0}" and "{1}" variables to the "dataV" library.'.format(vnm1, vnm2))
    return dataV

def SM_MovingAverage(datescol,datacol,mvwindow=7,byYear=True,aggregator=np.min):
    """SM_MovingAverage(datescol,datacol,mvwindow=7,byYear=True,aggregator=np.min)
      datescol:   column name of values in your "data" DataFrame
      datacol:    column name of dates in your "data" DataFrame
      mvwindow:   number of days for centered moving average window, default=7
      byYear:     extract yearly averages? default=True
      aggregator: numpy aggregation function, default=np.min
    Returns two new "dataV" variables.
    """
    # Refs:
    # http://stackoverflow.com/questions/19324453/add-missing-dates-to-pandas-dataframe
    # http://stackoverflow.com/questions/15771472/pandas-rolling-mean-by-time-interval

    tempSeries = pandas.Series(data[datacol].values.tolist(), index = pandas.DatetimeIndex(data[datescol]), name=datacol)
    idx = pandas.date_range(min(tempSeries.index), max(tempSeries.index))
    tempSeries = tempSeries.reindex(idx, fill_value=np.NaN)
    MASeries = pandas.rolling_mean(tempSeries, mvwindow, min_periods=mvwindow, center=True)
    if byYear:
        vnm1 = '_'.join([datacol, 'Year'])
        vnm2 = '_'.join([datacol, '{0}DayYearly{1}'.format(mvwindow, aggregator.func_name[1:].title())])
        vnm3 = '_'.join([datacol, '{0}DayYearly{1}Count'.format(mvwindow, aggregator.func_name[1:].title())])
        df = pandas.DataFrame({vnm2: MASeries, 'Year': pandas.Series([i.year for i in MASeries.index], index=MASeries.index)})
        byYear = df.groupby('Year')
        dataV[vnm1] = dict(list(byYear)).keys()
        dataV[vnm2] = byYear[vnm2].aggregate(aggregator).values.tolist()
        if raw_input('Do you want to save detailed MovingAverage data to a csv file? (y/[n])')=='y':
            cntnan = lambda grp: np.count_nonzero(~np.isnan(grp))
            vnm3c = byYear[vnm2].aggregate(cntnan).values.tolist()
            SaveOutputCSV([dataV[vnm1],dataV[vnm2],vnm3c], [vnm1,vnm2,vnm3], '{}-MovAvg'.format(datacol))
    else:
        vnm1 = '_'.join([datacol, 'Year'])
        vnm2 = '_'.join([datacol, '{0}DayMvAvg'.format(mvwindow)])
        dataV[vnm1] = [i for i in MASeries.index]
        dataV[vnm2] = MASeries.values.tolist()
    print('\nSuccessfully added the "{0}" and "{1}" variables to the "dataV" library\nNote: there may be NaN values in the "{1}" variable.'.format(vnm1, vnm2))
    return dataV

def SM_SampleCIs(col, citype="mean", conf=0.95):
    """SM_SampleCIs(col, citype="mean", conf=0.95); H&H 3.3 & 3.4
        returns either the mean or median
        with upper and lower confidence intervals.
        col:   a list/tuple of data, e.g. from Col2list
        citype: "mean" or "median"
        conf:   confidence level desired"""
    a = 1.0*np.array(col)
    n = len(a)
    if citype == 'mean':
        """http://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data"""
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t._ppf((1+conf)/2., n-1)
        out = n, m-h, m, m+h
    else:   # returns median and non-parametric confidence intervals
        """http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.binom.html"""
        citype = 'median'
        z = int(stats.binom.ppf((1.-conf)/2., n, 0.5))
        rl, ru = z-1, n-z-1
        out = n, sorted(a)[rl], np.median(a), sorted(a)[ru]
    citypep = {'mean':'Mean', 'median':'Median'}
    print('')
    for line in [('Sample-n', 'Lower {}% CI'.format(int(conf*100)), citypep[citype], 'Upper {}% CI'.format(int(conf*100))), out]:
        print('{0: ^12}{1: ^16}{2: ^16}{3: ^16}'.format(*line))

def SM_SamplePIs(col, pitype='p', conf=0.95, sides=2):
    """SM_SamplePIs(col, sides=2, pitype='np', conf=0.95); H&H 3.5, 3.6
        always returns (lower-pi, upper-pi) prediction intervals
        even if sides = 1. This lets you decide whether you want
        X > alpha or X < alpha, where alpha = 1 - conf.
        sides:  2 = both sides of pdf
        pitype: 'np' or 'p' for non-parametric or gaussian
        conf:    confidence level desired"""
    n = len(col)
    alpha = (1. - conf)/sides
    pitypep = {'p':'Parametric', 'np':'Nonparametric'}
    if pitype == 'np':
        from scipy.interpolate import interp1d
        y_interp = interp1d([float(i) for i in range(n)], sorted(col))
        zl, zu = alpha*n, (1. - alpha)*(n-1)
        out = pitypep[pitype], n, float(y_interp(zl)), float(y_interp(zu))
    else:   # pitype == 'p'
        stdev2 = np.std(col)**2
        h = stats.t.ppf(1. - alpha, n-1)*np.sqrt(stdev2+stdev2/n)
        out = pitypep[pitype], n, np.mean(col)-h, np.mean(col)+h
    print('')
    for line in [('PI type', 'Sample-n', 'Lower {}% PI'.format(int(conf*100)), 'Upper {}% PI'.format(int(conf*100))), out]:
        print('{0: ^16}{1: ^12}{2: ^16}{3: ^16}'.format(*line))

def SM_PercentileCIs(col, percentile, conf=0.95, sides=2):
    """SM_SampleCIs(col, citype="mean", conf=0.95); H&H 3.3 & 3.4
        returns either the mean or median
        with upper and lower confidence intervals.
        col:   a list/tuple of data, e.g. from Col2list
        citype: "mean" or "median"
        conf:   confidence level desired"""
    n = len(col)
    rl, ru = int(stats.binom.ppf((1.-conf)/sides, n, percentile)), int(stats.binom.ppf((1.+conf)/sides, n, percentile))-1
    from scipy.interpolate import interp1d
    y_interp = interp1d([float(i) for i in range(n)], sorted(col))
    out = n, '{0}% = {1}'.format(percentile*100., y_interp(percentile*(n+1)-1)), sorted(col)[rl], sorted(col)[ru]
    print('')
    for line in [('Sample-n', 'Percentile', 'Lower {}% CI'.format(int(conf*100)), 'Upper {}% CI'.format(int(conf*100))), out]:
        print('{0: ^12}{1: ^16}{2: ^16}{3: ^16}'.format(*line))

def SM_Bootstrap(indata,a):
    """SM_Bootstrap(indata,a), where a=alpha
        Tutorial:   http://www.randalolson.com/2012/08/06/statistical-analysis-made-easy-in-python/
        pyLibrary:  https://github.com/cgevans/scikits-bootstrap
        Data:       https://github.com/rhiever/ipython-notebook-workshop/blob/master/parasite_data.csv """
    try:    import scipy, scikits.bootstrap as bootstrap
    except: print("Error:", sys.exc_info()[1]); time.sleep(5);  return None
    CIs = bootstrap.ci(data=indata, statfunction=scipy.median, alpha=a)
    print("Bootstrapped "+str(100*(1-a))+"% confidence intervals\nLow:", CIs[0], "\tHigh:", CIs[1])

def SM_QSC(col):
    """SM_QSC(col)
    Pandas dataframe.describe() includes [-4]lower% [-3]50% [-2]upper%
    percentiles. Dif(upper,lower)/Dif(total) is a measure of quartile skew
    coefficient qs (Kenney and Keeping,1954). """
    try:    p=[dstats[:][col][i] for i in [-4,-3,-2]]; return ((p[2]-p[1])-(p[1]-p[0]))/(p[2]-p[0])
    except: print("Error:", sys.exc_info()[1]); time.sleep(5);  return None

def SM_CorrCoeffs(c1,c2):
    """SM_CorrCoeffs(c1,c2)
    Prints the correlation coefficients between two lists of measurements, c1 and c2 """
    return [stats.pearsonr(c1,c2)[0], stats.spearmanr(c1,c2)[0], stats.kendalltau(c1,c2)[0]]

def SM_RankSum(c1,c2,c1l,c2l,cont=False):
    """SM_RankSum(c1,c2,c1l,c2l,cont=False)
        Cont = False -
            Compute the Wilcoxon rank-sum statistic for two samples. The Wilcoxon rank-sum
            test tests the null hypothesis that two sets of measurements are drawn from the
            same distribution. The alternative hypothesis is that values in one sample are
            more likely to be larger than the values in the other sample. This test should
            be used to compare two samples from continuous distributions. It does not handle
            ties between measurements in x and y. Returns: z-statistic, assumes large-sample
            approximation and a normally distributed rank sum, and an adjusted 1-sided
            p-value of the test (i.e. 1/2 scipy's 2-sided p-value).
        Cont = True -
            Computes the Mann-Whitney rank test on samples x and y. Use only when the number
            of observation in each sample is > 20 and you have 2 independent samples of ranks.
            Mann-Whitney U is significant if the u-obtained is LESS THAN or equal to the
            critical value of U. This test corrects for ties and by default uses a continuity
            correction. The reported p-value is for a one-sided hypothesis, to get the
            two-sided p-value multiply the returned p-value by 2. Returns u, Mann-Whitney
            statistics, and prob, a one-sided p-value assuming an asymptotic normal
            distribution.
        Cont = exact -
            Computes the "exact version" of the rank sum test, as outlined in SMIWR chapters
            4 and 5 for sample sizes less than 10 data points each. Returns 1-sided p value. """
    try:
        import itertools
        if cont==False: return stats.ranksums(c1,c2)/2
        elif cont==True:   return stats.mannwhitneyu(c1,c2,use_continuity=True)
        else:
            if max(len(c1),len(c2)) > 10:   print('Table B4 does not support the exact rank sum test for data sets with more than 10 points'); return None
            elif len(c1) >= len(c2):  cm = c1; cn = c2; (cnl,cml)=(c2l,c1l)
            else:   cm = c2;    cn = c1; (cnl,cml)=(c1l,c2l)
            LoadTB(4)
            dup = {}; cnt = {}; rnk = {}; inc = 0
            a1 = np.array(sorted(cn+cm)); a2 = list(range(1,len(cn+cm)+1))
            # Modify a2 ranks for duplicate a1 values => a3
            for i in a1:    dup[i] = dup.get(i,0)+a2[inc]; cnt[i] = cnt.get(i,0)+1; inc += 1
            for i in cnt: rnk[i] = float(dup[i])/cnt[i]
            a3 = [rnk[i] for i in a1]
            # Compute the joint-rank probabilities
            b = [sum(i) for i in list(itertools.combinations(a3,2))]#; bu = np.unique(b)
            # Print distribution stats and plot
            print('Exact form of the Rank-Sum test:')
            print('Sample sizes:',', '.join([str(len(c1)), str(len(c2))]),'; \tRange of RankSums = ',' - '.join([str(b[0]),str(b[-1])]))
            Wsn = sum([a3[np.nonzero(a1==i)[0][0]] for i in cn]); print('Exp[RankSum] in '+cnl+' =', Wsn,)
            Wsm = sum([a3[np.nonzero(a1==i)[0][0]] for i in cm]); print('; \tExp[RankSum] in '+cml+' =', Wsm)
            if Wsn < Wsm:   Pcn = Lookupx(4,len(cn),len(cm),Wsn,'xs'); tc = cnl+' < '+cml
            else:   Pcn = Lookupx(4,len(cn),len(cm),Wsn,'x'); tc = cnl+' > '+cml
            print('Prob ['+tc+'] =',1-Pcn,'; \t\tProb ['+cnl+' != '+cml+'] =', 1-2*Pcn)
            return Pcn
    except: print("Error:", sys.exc_info()[1]); time.sleep(5);  return None

def SM_SignedRank(c1,c2):
    """SM_SignedRank(c1,c2)
    Exact form of the Wilcoxon signed-ranks test (Chpt 6, SMIWR) """
    LoadTB(6)
    if len(c1) > len(c2):  cn = c2; cm = c1
    else:   cn = c1; cm = c2
    d = np.array([(abs(cn[i]-cm[i]),cmp(cn[i]-cm[i],0),i+1) for i in range(len(cn)) if cn[i]!=cm[i]])
    ds = d[np.lexsort((d[:,0],d[:,0]))]; dup = {}; cnt = {}; rnk = {}
    for i in range(len(ds)): ds[i,2]=i+1; dup[ds[i,0]] = dup.get(ds[i,0],0)+ds[i,2]; cnt[ds[i,0]] = cnt.get(ds[i,0],0)+1
    for i in cnt: rnk[i] = float(dup[i])/cnt[i]
    for i in range(len(ds)):    ds[i,2] = rnk[ds[i,0]]
    Ws = sum([ds[i,2] for i in range(len(ds)) if ds[i,1]>0])
    if Ws > 3*len(cn):
        alpha = Lookupx(6,1,len(cn),Ws,'')
        print('alpha [W+ >= '+str(Ws)+'] =', alpha)
    else:
        alpha = Lookupx(6,-1,len(cn),Ws,'')
        print('alpha[W+ <= '+str(Ws)+'] =', alpha)
    return Ws, alpha

def SM_SLRplot(x, y, xl='x', yl='y', lstats=None, plot='pylab.show()'):
    """SM_SLRplot(x, y, xl='x', yl='y', lstats=None, plot='pylab.show()')
    Returns list of slope, intercept, r_value, p_value, std_err. """
    if lstats==None:   lstats = stats.linregress(x,y)
    fitl = np.poly1d(lstats[0:2])
    print('%s = %f * %s + %f, R^2 = %f, p = %f' % tuple([yl,lstats[0],xl]+list(lstats[1:4])))
    fig = pylab.figure(); fig.patch.set_facecolor('white')
    fig.canvas.set_window_title('SLR Plot: y = '+str(lstats[0])+'*x +'+str(lstats[1])+'; (Close to continue...)')
    pylab.plot(x,y,'bx',[min(x),max(x)],[fitl(min(x)),fitl(max(x))],'k--'); ax = pylab.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pylab.grid(b=True, which='major', color='k', linestyle=':')
    pylab.grid(b=True, which='minor', color='g', linestyle=':')
    pylab.xlabel(xl); pylab.ylabel(yl)
    eval(plot)

def SM_SLRConfIntervals(xd, yd, xl='x', yl='y', lstats=None, conf=0.95, x=None):
    """SM_SLRConfIntervals(xd, yd, xl='x', yl='y', lstats=None, conf=0.95, x=None)
    http://astropython.blogspot.com/2011/12/calculating-and-plotting-prediction.html
    Calculates the confidence band of the linear regression model at the
    desired confidence level. The 2sigma confidence interval is 95% sure to
    contain the best-fit regression line. This is not the same as saying it will
    contain 95% of the data points.
    Arguments:
        - conf:     desired confidence level, by default 0.95 (2 sigma)
        - xd,yd:    Numpy data arrays
        - a,b:      (optional) SLR constants, as in y=ax+b
        - x:        (optional) array with x values to calculate the confidence
                    band. If none is provided, will by default generate 100
                    points in the original x-range of the data.
    Calculates lcb,ucb,x,y: arrays holding the lower and upper confidence bands
                    corresponding to the [input] x array.
    Plots a shaded area containing the confidence band
    References:
        1. http://en.wikipedia.org/wiki/Simple_linear_regression,
            see Section Confidence intervals
        2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm
    Rodrigo Nemmen v1 Dec. 2011 v2 Jun. 2012: corrected bug in computing dy """
    def scatterfit(x, y, a=None, b=None):
        """scatterfit(x, y, a=None, b=None)
        Compute the mean deviation of the data about the linear model given A,B
        (y=ax+b) provided as arguments.
        Otherwise, compute the mean deviation about the best-fit line.
        x,y assumed to be Numpy arrays.
        a,b SLR scalars.
        Returns the float sd with the mean deviation.
        Author: Rodrigo Nemmen """
        #if a==None: a, b, r, p, err = stats.linregress(x,y)
        # Std. deviation of an individual measurement (Bevington, eq. 6.15)
        N=np.size(x)
        sd=1./(N-2.)* np.sum((y-a*x-b)**2); sd=np.sqrt(sd)
        return sd
    xd = np.array(xd)
    yd = np.array(yd)
    alpha = 1.-conf # significance
    n = xd.size # data sample size
    if x == None: x = np.linspace(xd.min(),xd.max(),100) # Predicted values (best-fit model)
    if lstats==None: a, b, r, p, err = stats.linregress(xd,yd)
    else:   a=lstats[0]; b=lstats[1]
    y = a*x+b # Auxiliary definitions
    sd = scatterfit(xd,yd,a,b) # Scatter of data about the model
    sxd = np.sum((xd-xd.mean())**2)
    sx = (x-xd.mean())**2 # array # Quantile of Student's t distribution for p=1-alpha/2
    q = stats.t.ppf(1.-alpha/2.,n-2) # Confidence band
    dy = q*sd*np.sqrt( 1./n + sx/sxd )
    ucb = y+dy # Upper confidence band
    lcb = y-dy # Lower confidence band
    SM_SLRplot(xd, yd, xl, yl, lstats=lstats, plot='None')
    pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
    print('Confidence intervals: ', str(conf*100)+'%')
    pylab.show()
    #return lcb, ucb, x, y

def SM_TheilLine(x,y, sample= "auto", n_samples = 1e7):
    """SM_TheilLine(x,y, sample= "auto", n_samples = 1e7)
    Adapted from: https://github.com/CamDavidsonPilon/Python-Numerics/blob/master/Estimators/theil_sen.py
    Computes the Theil-Sen estimator for 2d data.
    PARAMETERS:
    x: 1-d np array, the control variate
    y: 1-d np.array, the ind variate.
    sample: if n>100, the performance can be worse, so we sample n_samples.
            Set to False to not sample.
    n_samples: how many points to sample.

    This complexity is O(n**2), which can be poor for large n. We will perform a
    sampling of data points to get an unbiased, but larger variance estimator.
    The sampling will be done by picking two points at random, and computing the
    slope, up to n_samples times.
    """
    import itertools
    x = np.array(x)
    y = np.array(y)
    def slope(x_1, x_2, y_1, y_2):  return (1 - 2*(x_1>x_2))*((y_2 - y_1)/np.abs((x_2-x_1)))
    assert x.shape[0] == y.shape[0], "x and y must be the same shape."
    n = x.shape[0]
    if n < 100 or not sample:
        ix = np.argsort(x)
        slopes = np.empty(n*(n-1)*0.5)
        for c, pair in enumerate(itertools.combinations(range(n),2)): #it creates range(n) =(
            i,j = ix[pair[0]], ix[pair[1]]
            slopes[c] = slope(x[i], x[j], y[i],y[j])
    else:
        i1 = np.random.randint(0, n, n_samples)
        i2 = np.random.randint(0, n, n_samples)
        slopes = slope(x[i1], x[i2], y[i1], y[i2])
        #pdb.set_trace()
    slope_ = stats.nanmedian(slopes)
    #find the optimal b as the median of y_i - slope*x_i
    intercepts = np.empty(n)
    for c in xrange(n):
        intercepts[c] = y[c] - slope_*x[c]
    intercept_ = scipy.median(intercepts)
    print('Kendall-Theil line:\t\tm: {0}, b:{1}'.format(slope_, intercept_))
    fig = pylab.figure(); fig.patch.set_facecolor('white')
    fig.canvas.set_window_title('Kendall-Theil Line: y = {0}*x + {1}; (Close to continue...)'.format(slope_, intercept_))
    pylab.plot(x,y,'bx',x,[slope_*xi+intercept_ for xi in x],'k--'); ax = pylab.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pylab.grid(b=True, which='major', color='k', linestyle=':')
    pylab.grid(b=True, which='minor', color='g', linestyle=':')
    pylab.xlabel('x'); pylab.ylabel('y')
    pylab.show()
    tls = stats.pearsonr(slope_*x+intercept_, y)
    return [slope_, intercept_]+list(tls)

def SM_QuantilePlot(cols=[], cnames=[], same=False):
    """SM_QuantilePlot(cols=[], cnames=[], same=False) creates a quantile plot(s)
    cols =    list of data lists,
    cnames =  column names for cols,
    same =    True for same figure, False for seperate figures.
    Based on: http://stackoverflow.com/questions/13865596/quantile-quantile-plot-using-scipy
              http://docs.scipy.org/doc/scipy/reference/stats.html """
    #data = np.random.normal(loc = 20, scale = 5, size=100)
    dn1 = raw_input("Which distribution: 0=norm, 1=lognorm, 2=expon, 3=powerlaw, 4=gumbel_r, 5=gumbel_l:")
    dst = ['norm','lognorm','expon','powerlaw','gumbel_r','gumbel_l']
    if not dn1:  dn2 = dst[0]   # default(none) = norm
    else:   dn2= dst[int(dn1)]  # otherwise = translate selection
    def QuantPlotID():    # Plot method for a selected ID in the idf field.
        if isinstance(cols[0], list) and not isinstance(cols[0][0],float):  pass
        elif not isinstance(cols[0], list) and not isinstance(cols[0],float):  pass
        elif dn2 == 'lognorm':    # For lognormal, ignore values <= 0
            if isinstance(cols[0], list):   ds = [scipy.log(i) for i in cols]; dn3='norm'
            else:   ds = scipy.log(cols); dn3='norm'
        else:   ds = cols; dn3=dn2
        try:
            fig = pylab.figure(); fig.patch.set_facecolor('white')
            fig.canvas.set_window_title('Probability Plot for '+', '.join(cnames))
            osmr, ps, = stats.probplot(ds, dist=dn3, plot=pylab); ax = pylab.gca()
            pylab.xlabel(dn2+" Quantiles"); pylab.ylabel("Sorted, "+dn2+" Measurements")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            pylab.grid(b=True, which='major', color='k', linestyle=':')
            pylab.grid(b=True, which='minor', color='g', linestyle=':')
        except:
            print("Error: the", dn2,"distribution does not fit the data.", sys.exc_info()[1]); time.sleep(5);  return None

    if same:    # If 2 columns were specified for the same QQ plot.
        try:
            p = []
            fig = pylab.figure(); fig.patch.set_facecolor('white')
            fig.canvas.set_window_title('Probability Plots for '+', '.join([cnms[i] for i in cols])+'; (Close to continue...)')
            for i in range(len(cols)):
                if dn2 == 'lognorm':
                    ds = scipy.log([val for val in cols[i] if val > 0.]); dn3='norm'
                else:   ds = cols[i]; dn3=dn2
                ps[i] = stats.probplot(ds, dist=dn3, plot=pylab)
            ax = pylab.gca()
            pylab.xlabel(dn2+" Quantiles"); pylab.ylabel("Sorted, "+dn2+" Measurements")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            pylab.grid(b=True, which='major', color='k', linestyle=':')
            pylab.grid(b=True, which='minor', color='g', linestyle=':')
            pylab.legend(list(' '.join(cnames))+[' '], loc='best')
            print('PPCC_norms:\n', ',\t'.join(cnames)+'\n', ',\t'.join([str(round(val[-1][-1],4)) for val in ps]))
        except:
            print("Error:", sys.exc_info()[1]); time.sleep(5);  return None
        pylab.show()
    else:       # Plot seperate QQ plots for all columns in cols.
        rs = []
        for i,cols in enumerate(cols):
            ps = QuantPlotID(cnames[i])
            rs.append(round(ps[-1],4))
        print('R_norms:\n'+',\t'.join(cnames)+'\n'+',\t'.join([str(k) for k in rs]))
        pylab.show()

def SM_PlotQQ(c1,c2,c1l,c2l):
    """ SM_PlotQQ(c1,c2,c1l,c2l) plots data columns, c1 vs. c2 after ordering
    the values like quantiles. """
    def linearfit(x,y):
        def linear_errors(m,x,y):
            return m*x - y
        from scipy.optimize import leastsq
        m = leastsq( linear_errors, y[-1]/x[-1], args=(x,y) )
        return m[0]
    fig = pylab.figure(); fig.patch.set_facecolor('white')
    fig.canvas.set_window_title(c1l+' vs. '+c2l+' (close this window to continue)')
    sc1 = sorted(c1); sc2 = sorted(c2); a=[]
    fitr = linearfit(sc1,sc2); fitl = np.poly1d([fitr,0])
    pylab.plot(sc1,sc2,'bx',[0,max(sc1)],fitl([0,max(sc1)]),'g--', linewidth=2.0); ax = pylab.gca()
    pylab.figtext(0.7, 0.925, 'm: '+str(fitr[0]))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pylab.grid(b=True, which='major', color='k', linestyle=':')
    pylab.grid(b=True, which='minor', color='g', linestyle=':')
    pylab.xlabel('Sorted '+c1l); pylab.ylabel('Sorted '+c2l)
    pylab.show()

def SM_BoxPlot(cols,xlbls,ylbl,ttl, idf=-1):
    """" SM_BoxPlot(cols,xlbls,ylbls,ttl,idf=-1) plots a box plot of cols with
    an optional idf query of the ID field.
    http://matplotlib.org/examples/pylab_examples/boxplot_demo2.html """
    def BoxPlotID(idq):    # Plot method for a selected ID in the idf field.
        ds = Col2list(cols[0],fil=[idf,"=='"+idq+"'"])
        fig = pylab.figure(); fig.patch.set_facecolor('white')
        fig.canvas.set_window_title(idq+' '+ttl)
        pylab.boxplot(ds)
        pylab.grid(b=True, which='major', color='k', linestyle=':')
        pylab.grid(b=True, which='minor', color='g', linestyle=':')
        pylab.ylabel(idq+' '+ylbl)

    if idf < 0:         # Plot method for the list of 'cols' in the csv file.
        fig = pylab.figure(); fig.patch.set_facecolor('white')
        fig.canvas.set_window_title(ttl)
        pylab.boxplot(cols)
        ax = pylab.gca(); #ax.yaxis.set_minor_locator(AutoMinorLocator())
        pylab.grid(b=True, which='major', color='k', linestyle=':')
        pylab.grid(b=True, which='minor', color='g', linestyle=':')
        xtN = pylab.setp(ax, xticklabels=xlbls); pylab.setp(xtN, rotation=15)
        pylab.ylabel(ylbl)
        pylab.show()
    else:               # Calls ListBox function to plot IDs in the idf field.
        idlist = sorted(UniqueIDs2List(idf))
        ListBoxSelectNRun(idlist, BoxPlotID, 'BoxPlot', ExtraCommand='pylab.show()')

def SM_HodgesLehmannDiff(c1,c2):
    import itertools
    """ Hodges-Lehmann Estimation of the magnitude of difference between two groups
    (Chpt 6, SMIWR). Returns the median delta estimator, a median unbiased estimator
    of the difference in the medians of non-normal populations x and y. A shift of
    size median delta makes the data appear devoid of any evidence of difference
    between x and y when viewed by the rank-sum test. """
    d = sorted([i-j for (i,j) in itertools.product(c1,c2)])
    m = scipy.median(d); z = stats.mstats.zscore(d); p = stats.norm.sf(z)
    print('Median delta ('+cnms[1]+' - '+cnms[2]+') =', m)
    #SM_Bootstrap(d,0.05)
    fig = pylab.figure(); fig.patch.set_facecolor('white')
    fig.canvas.set_window_title('Hodges-Lehmann Plot for dif(cn,cm)-vs-p; (Close to continue...)')
    pylab.plot(d,p,'b+-',[m,m],[min(p),max(p)],'r-')
    ax = pylab.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pylab.grid(b=True, which='major', color='k', linestyle=':')
    pylab.grid(b=True, which='minor', color='g', linestyle=':')
    pylab.xlabel('dif(c1,c2)'); pylab.ylabel('p')
    pylab.legend(['diff(cn,cm)','H-L Median'],'best')
    pylab.show()
    return m

def SM_MannKindall_test(x, alpha = 0.5):
    """
    This performs the MK (Mann-Kendall) test to check if the trend is present in
    the x (residuals) or not, based on the specified alpha level.
    Source: http://www.ambhas.com/codes/statlib.py
    Input:
        x:      a vector of data
        alpha:  significance level
    Output:
        h:      True (if trend is present) or False (if trend is absence)
        p:      p value of the significance test
    Example:
        >>> x = np.random.rand(100)
        >>> h,p = mk_test(x,0.05)  # meteo.dat comma delimited
    """
    from scipy.stats import norm
    x = np.array(x)
    n = len(x)
    # calculate S
    s = 0
    for k in xrange(n-1):
        for j in xrange(k+1,n):
            s += np.sign(x[j] - x[k])
    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)
    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in xrange(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18
    if s>0:         z = (s - 1)/np.sqrt(var_s)
    elif s == 0:    z = 0
    elif s<0:       z = (s + 1)/np.sqrt(var_s)
    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z))) # two tail test
    h = abs(z) > norm.ppf(1-alpha/2)
    print('Mann-Kendall: Trend=%s, alpha=%f, p=%f, z=%f, S=%f' % (str(h),alpha,p,z,s))
    #return h, p, s, z

def SM_DurbinWatsonSerialCor(e):
    """http://www.xyzhang.info/implement-durbin-watson-auto-correlation-test-in-python/
    The Durbin-Watson Serial Correlation value, d, varies from 0 to 4.
    d >> 2, is evidence of positive serial correlation
    d ~ 2 indicates very weak or no autocorrelation
    d << 2, is evidence of negative serial correlation

    To test for positive autocorrelation at significance a, the test statistic
    d is compared to lower and upper critical values (dL,a and dU,a):

    d < dL,a, is evidence of positive autocorrelation.
    d > dU,a, means no statistical evidence of positive autocorrelation.
    dL,a < d < dU,a, means the test is inconclusive.
    """
    import math
    d = sum([ math.pow(e[i+1] - e[i],2)  for i in range(len(e)-1)]) / sum([ math.pow(i,2) for i in e ])
    print('Durbin-Watson: d=%f, (d-2)/2=%s' % (d, str(round((d-2.)*100.0/2.,1))+'%'))

def SM_1WayAnova(*args):
    """ All lists passed to this function are passed to Scipy's 1-way ANOVA. Returns (F-value, p-value).
    Doc: http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html """
    return stats.f_oneway(*args)

def SM_ExceedancePlot(col, exceedtype='Cunnane', vnm = 'x'):
    """ SM_ExceedancePlot(col, exceedtype='Cunnane', vnm = 'x')
    Ref: H&H 2.1.3 "Quantile Plots"

    http://python-scipy.sourcearchive.com/documentation/0.7.0-2/namespacescipy_1_1stats_1_1mstats__basic_ac1facaf57c97ee82d89cb9bd5e380b83.html
    stats.mstats_basic.plotting_positions(data, alpha = 0.4, beta = 0.4)
        Plotting positions are defined as (i-alpha)/(n-alpha-beta), where:
            - i is the rank order statistics
            - n is the number of unmasked values along the given axis
            - alpha and beta are two parameters.

        Typical values for alpha and beta are:
            - (0,1)    : *p(k) = k/n* : linear interpolation of cdf (R, type 4)
            - (.5,.5)  : *p(k) = (k-1/2.)/n* : piecewise linear function (R, type 5)
            - (0,0)    : *p(k) = k/(n+1)* : Weibull (R type 6)
            - (1,1)    : *p(k) = (k-1)/(n-1)*. In this case, p(k) = mode[F(x[k])].
              That's R default (R type 7)
            - (1/3,1/3): *p(k) = (k-1/3)/(n+1/3)*. Then p(k) ~ median[F(x[k])].
              The resulting quantile estimates are approximately median-unbiased
              regardless of the distribution of x. (R type 8)
            - (3/8,3/8): *p(k) = (k-3/8)/(n+1/4)*. Blom.
              The resulting quantile estimates are approximately unbiased
              if x is normally distributed (R type 9)
            - (.4,.4)  : approximately quantile unbiased (Cunnane)
            - (.35,.35): APL, used with PWM
    """
    sdata = sorted(col)
    alphabeta = {'Cunnane':  {'alpha':0.4, 'beta':0.4},
                 'R type 4': {'alpha':0., 'beta':1},
                 'R type 5': {'alpha':0.5, 'beta':0.5},
                 'Weibull':  {'alpha':0, 'beta':0},
                 'R type 7': {'alpha':1, 'beta':1},
                 'R type 8': {'alpha':1/3., 'beta':1/3.},
                 'R type 9': {'alpha':3/8., 'beta':3/8.},
                 'APL':      {'alpha':0.35, 'beta':0.35}}
    exceed = [i*100. for i in stats.mstats_basic.plotting_positions(sdata, **alphabeta[exceedtype])]
    if raw_input('Do you want to write the exceedance data to file? (y/n)')=='y':
        SaveOutputCSV([sdata,exceed],[vnm,'P_exceedance'],'{}-exceedance'.format(vnm))
    fig = pylab.figure(); fig.patch.set_facecolor('white')
    fig.canvas.set_window_title('Exceedance Plot for {0}; (Close to continue...)'.format(vnm))
    pylab.plot(sdata, exceed); ax = pylab.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pylab.grid(b=True, which='major', color='k', linestyle=':')
    pylab.grid(b=True, which='minor', color='g', linestyle=':')
    pylab.ylabel('Exceedance Frequency (%)'); pylab.xlabel(vnm)
    pylab.show()
    return sdata, exceed

def SM_RalphPlot(dates, flows, flowmax, vnm='x'):
    """ SM_RalphPlot(dates, flows, flowmax) plots the number of times list(flows)
    exceeds float(flowmax) versus year in list(dates).
    Script expects dates in units of year+month/12. format."""
    from collections import defaultdict
    dateyr = [int(float(yr)-0.001) for yr in dates]
    d = defaultdict(int)
    for i,yr in enumerate(dateyr):
        if flows[i] > flowmax:  d[int(yr)] += 1
    [yrs, cnts] = zip(*sorted(d.items()))
    yrs = list(yrs)
    cnts = list(cnts)
    fnout = str(fname).replace('.', '_RALPH_{0}_gt_{1}-cfs.'.format(vnm, str(flowmax).replace('.','p')))
    np.savetxt(fnout, np.column_stack([yrs, cnts]), fmt='%i', delimiter=',')
    for i, line in enumerate(fileinput.input(fnout, inplace = 1)):
        if i == 0:  sys.stdout.write('Year,#_gt_{0}-cfs\n'.format(flowmax))
        sys.stdout.write(line)
    print('Wrote:', fnout+'. Closing.')
    fig = pylab.figure(); fig.patch.set_facecolor('white')
    fig.canvas.set_window_title('RALPH Plot for {0} exceeding {1} cfs; (Close to continue...)'.format(vnm, flowmax))
    pylab.plot(yrs, cnts); ax = pylab.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pylab.grid(b=True, which='major', color='k', linestyle=':')
    pylab.grid(b=True, which='minor', color='g', linestyle=':')
    pylab.xlabel('Year'); pylab.ylabel('Total # > {0} cfs'.format(flowmax))
    pylab.show()

if __name__ == '__main__':
    SMmethods = sorted([v for v in globals().keys() if (v.startswith(('SM','Add')))])
    print('Available methods include:\n {}.'.format(', '.join(SMmethods)))
    fname, fpath, fnm1 = SelectFile('Please select a csv/txt file with your columns of measurements:')
    os.chdir(fpath)
    try:    data = pandas.read_csv(fname, na_values=[""," ",'None'])
    except: print("Error:", sys.exc_info()[1]); time.sleep(5);  sys.exit()
    cnms = sorted(data.columns.tolist())
    print('\nYour original variable names include:\n{}'.format(', '.join(cnms)))
    if raw_input("Do you want to remove the null values first? (y/n)")=='y':
        data = data.dropna(axis=0)
        print('\nYou chose to drop null values.\n')
    dtcols = raw_input('If there are any dates/times columns in this file,\nplease enter which ones:\n(e.g. Enter "1,2" for the columns 1 and 2.)')
    if dtcols:
        for col in (int(i)-1 for i in dtcols.split(',')):
            data[cnms[col]] = data[cnms[col]].apply(parsedate)
    # Describe the data
    dstats = data.describe()
    print(fnm1+" descriptions:\n", dstats, '\n', 'QSC:',''.join(['%12s' % round(SM_QSC(col),6) for col in cnms if not isinstance(data[col][0],(str,datetime.date))])+'\n')
    dataV = {}
    # Select tool(s) and input to perform analysis
    FuncSelectRun(SMmethods)

    #print('\nExpected values for S2321500:')
    #for line in [('Stat', 'Mean', 'Median'),('-'*12, '-'*16, '-'*16)]+zip(('ExpValue', 'lower95%CI', 'upper95%CI'), SM_SampleCIs(Col2list('S2321500')), SM_SampleCIs(Col2list('S2321500'), citype='median')):
    #    print('{0: ^12}{1: ^16}{2: ^16}'.format(*line))
    #print('\nPrediction intervals for S2321500:')
    #for line in [('Stat', 'Parametric', 'Non-parametric'),('-'*16, '-'*16, '-'*16)]+zip(('lower95%PI', 'upper95%PI'), SM_SamplePIs(Col2list('S2321500')), SM_SamplePIs(Col2list('S2321500'), pitype='np')):
    #    print('{0: ^16}{1: ^16}{2: ^16}'.format(*line))
    #SM_Bootstrap()
    #SM_QuantilePlot([1,2])
    #SM_AddDiffCol('FW-WS',0,2)
    #print cnms[1]+'-vs-'+cnms[1]+': R, Rho, tau =','\t'.join([str(i) for i in SM_CorrCoeffs(Col2list(1),Col2list(2))])
    #SM_SLRplot(Col2list(1),Col2list(2),cnms[1],cnms[2])
    #SM_slrconf(scipy.log10(Col2list(1)),scipy.log10(Col2list(2)),cnms[1],cnms[2],0.975)
    #print('Rank Sum: p['+cnms[1]+' < '+cnms[2]+'] =', 1-SM_RankSum(scipy.log(Col2list(1,[1,'>0'])),scipy.log(Col2list(2,[2,'>0'])),cnms[1],cnms[2])[1])
    #print('Rank Sum: p['+cnms[1]+' < '+cnms[2]+'] =', 1-SM_RankSum(Col2list(1),Col2list(2),cnms[1],cnms[2])[1], '\n')
    #print('Mann-Whitney: p['+cnms[1]+' < '+cnms[2]+'] =', 1-SM_RankSum(scipy.log10(Col2list(1,[1,'>0'])),scipy.log10(Col2list(2,[2,'>0'])),cnms[1],cnms[2],cont=True)[1])
    #print('Mann-Whitney: p['+cnms[1]+' < '+cnms[2]+'] =', 1-SM_RankSum(Col2list(1),Col2list(2),cnms[1],cnms[2],cont=True)[1], '\n')
    #SM_RankSum(Col2list(1)[0:10],Col2list(2)[0:10],cnms[1],cnms[2],'exact')
    #SM_SignedRank(scipy.log10(Col2list(1)[0:10]),scipy.log10(Col2list(2)[0:10]))
    #SM_QuantilePlot([1,2],same=True)
    #SM_QuantilePlot([1,2])
    #SM_QuantilePlot(cols=[4],idf=0)
    #SM_PlotQQ(Col2list(1),Col2list(2),cnms[1],cnms[2])
    #SM_BoxPlot([scipy.log10(Col2list(1)),scipy.log10(Col2list(2))],[cnms[1],cnms[2]],'Log (Flow), cfs','Flow BoxPlot')
    #SM_BoxPlot([4],[cnms[0]],'Well level, ft','BoxPlot', idf=0)
    #SM_HodgesLehmannDiff(Col2list(1),Col2list(2))
    #SM_SLRConfIntervals(Col2list(0),Col2list(1),xl=cnms[0],yl=cnms[1],conf=0.95)
    #tl = SM_TheilLine(Col2list(0),Col2list(1))
    #print('Theil line: m=%f, b=%f' % tuple(tl[:2]))
    #SM_SLRConfIntervals(Col2list(0),Col2list(1),xl=cnms[0],yl=cnms[1],lstats=tl,conf=0.95)
    #SM_MannKindall_test(Col2list(1),alpha=0.05)
    #SM_DurbinWatsonSerialCor(Col2list(1))
    #print('1-way ANOVA: F = %f, p = %f' % SM_1WayAnova(Col2list(0),Col2list(1)))
    #
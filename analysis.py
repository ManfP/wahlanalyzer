import pandas as pd
import geopandas
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import contextily as ctx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import tabulate
import requests
import os
import shutil
from jinja2 import Environment, FileSystemLoader, select_autoescape
from collections import Counter

WAHL_NAME = "Kommunalwahl Bonn 2020 (Stadtrat)"
OUTDIR = "out/"
parties = ["CDU","SPD","GRÜNE","FDP","LINKE","BBB","AfD","PIRATEN","BIG","PARTEI","S. Post","Volt"]
columns = ["datum","wahl","ags","gebiet_nr","gebiet_name","max_schnellmeldungen","schnellmeldungen","nicht_sperr_w","sperr_w","nicht_verzeichnis","berechtigte","wähler","wähler_wahlschein","ungültig","gültig"]+parties
party_colors = {"GRÜNE":"green","SPD":"red","CDU":"black","BBB":"blue","BIG":"yellow"} # only needed for winning parties in each district. matplotlib colors
URL_RESULTS = "http://wahlen.bonn.de/wahlen/KW2020/05314000/html5/Open-Data-Ratswahl-NRW436.csv"
#URL_RESULTS = "Open-Data-Ratswahl-NRW436.csv" # this can be a local file too
URL_GEODATA = "https://stadtplan.bonn.de/geojson?OD=4450"
FILE_GEODATA = "geo.json"
PLOT_DPI = 150
PCA_COMPONENTS_TABLE = 5
PCA_COMPONENTS_MAPS = 3

# The following should match each row in results to geodata
INDEX_RESULTS = "gebiet_nr"
INDEX_GEODATA = "stimmbezirk_corona"


if os.path.exists(OUTDIR):
	shutil.rmtree(OUTDIR)
shutil.copytree("static", OUTDIR)

if not os.path.exists(FILE_GEODATA):
	s = requests.get(URL_GEODATA).text
	with open(FILE_GEODATA, "w") as f:
		f.write(s)

jinja_env = Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape(["html"]))	

geodata = geopandas.read_file(FILE_GEODATA).to_crs("epsg:3857")
results = pd.read_csv(URL_RESULTS, sep = ";").fillna(0)
assert len(results.columns) == len(columns)
results.columns = columns
data = geopandas.GeoDataFrame(results).set_index(INDEX_RESULTS).join(geodata.set_index(INDEX_GEODATA)["geometry"]) # Results + Geometry
data_lokal = data[data.berechtigte != 0] # filter out ze briefwahlbezirke
briefwahl_per_w = data[data.berechtigte == 0].wähler.sum() / data[data.berechtigte != 0].sperr_w.sum() # Briefwähler / Wähler mit W-Vermerk

def make_map(d, outfn, **kwargs):
	fig = plt.figure(dpi = PLOT_DPI, frameon = False)
	ax = data_lokal.plot(d, alpha=0.5, legend=True, figsize=(12,12), **kwargs)
	ax.set_axis_off()
	ax.set_aspect('equal')
	ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom = 13)
	plt.box(False)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTDIR, outfn))
	plt.close(fig)
	



def do_pca():
	pca = PCA(whiten=True)
	rodat = []
	rodat.append(np.array((data_lokal.berechtigte-data_lokal.sperr_w-data_lokal.wähler)/data_lokal.berechtigte))
	rodat.append(np.array(data_lokal.sperr_w/data_lokal.berechtigte))
	rodat.append(np.array(data_lokal.ungültig/data_lokal.wähler))
	for i in parties:
		rodat.append(np.array(data_lokal[i]/data_lokal.gültig))
	rodat = np.array(rodat).transpose()
	rodat = StandardScaler().fit_transform(rodat)

	dat_pca = pca.fit_transform(rodat)
	return pca, dat_pca
	
pca, dat_pca = do_pca()
comps = pca.components_
dat_comps = dat_pca.transpose()

def get_pca_table():
	table = []
	for i in range(len(parties)+2):
		s = (["(nichtwähler)","(briefwahl)","(ungültig)"]+parties)[i]
		l = [s]
		l += [format(k[i], ".03f") for k in comps[:PCA_COMPONENTS_TABLE]]
		table.append(l)
	headers=["Partei"]+[f"Komp. {i+1}" for i in range(PCA_COMPONENTS_TABLE)]
	return tabulate.tabulate(table, headers=headers, tablefmt="html")
	
brief_gesamt=data[data.berechtigte==0].sum()
lokal_gesamt=data[data.berechtigte!=0].sum()
gesamt=data.sum()
def get_result_table():
	table = []
	for p in parties:
		l=[p]
		l.append(format(lokal_gesamt[p]/lokal_gesamt.gültig,".2%"))
		l.append(format(brief_gesamt[p]/brief_gesamt.gültig,".2%"))
		l.append(format(gesamt[p]/gesamt.gültig,".2%"))
		table.append(l)
	return tabulate.tabulate(table,headers=["Partei","Lokal","Brief","Gesamt"],tablefmt="html")

def make_pca_parties(outfn):
	fig = plt.figure(dpi=PLOT_DPI)
	X = []
	Y = []
	for i,p in enumerate(["(nichtwähler)","(briefwahl)","(ungültig)"]+parties):
		x=comps[0][i]
		y=comps[1][i]
		X.append(x)
		Y.append(y)
		plt.annotate(p, xy=(x,y))
	plt.plot(X,Y,".")
	plt.xlabel("Komponente 1")
	plt.ylabel("Komponente 2")
	plt.title("Parteien")
	plt.savefig(os.path.join(OUTDIR, outfn))
	plt.close(fig)
	
def make_pca_places(outfn):
	fig = plt.figure(dpi=PLOT_DPI)
	recs = data_lokal.to_records()
	assert len(recs)==len(dat_pca)
	leg = []
	for p in parties:
		l = [i for i in range(len(recs)) if max(parties,key=lambda x:recs[i][x])==p]
		if not l:
		    continue
		X=[dat_pca[i][0] for i in l]
		Y=[dat_pca[i][1] for i in l]
		plt.plot(X,Y,".",color=party_colors[p])
		leg.append(p)
	plt.title("Stimmbezirke nach stärkster Kraft")
	plt.xlabel("Komponente 1")
	plt.ylabel("Komponente 2")
	plt.legend(leg)
	plt.savefig(os.path.join(OUTDIR, outfn),dpi=PLOT_DPI)
	plt.close(fig)
	
district_winners = [max(parties,key=lambda i:d[i]) for d in data_lokal.to_records()]
winner_counts = Counter(district_winners)
def make_winner_map(outfn):
	fig = plt.figure(dpi = PLOT_DPI, frameon = False)
	ax = data_lokal.plot(None, alpha=0.5, legend=True, figsize=(12,12), color = [party_colors[i] for i in district_winners])
	ax.set_axis_off()
	ax.set_aspect('equal')
	ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom = 13)
	handles = []
	for i in sorted(winner_counts.keys(), key=winner_counts.get, reverse=True):
		handles.append(mpatches.Patch(color=party_colors[i], label=i))
	plt.legend(handles=handles)
	plt.box(False)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTDIR, outfn))
	plt.close(fig)
	

def make_corr_matrix(outfn):
	df = pd.DataFrame()
	df["Lokalwahl"] = data_lokal.wähler/data_lokal.berechtigte
	df["Briefwahl"] = data_lokal.sperr_w/data_lokal.berechtigte
	df["Ungültig"] = data_lokal.ungültig/data_lokal.wähler
	for p in parties:
		df[p] = data_lokal[p] / data_lokal.wähler
	fig = plt.figure(dpi = PLOT_DPI, frameon = False)

	f = plt.figure(figsize=(19, 15))
	plt.matshow(df.corr(), fignum=f.number)
	plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
	plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=14)
	plt.savefig(os.path.join(OUTDIR, outfn))
	plt.close(fig)
	
def get_corr_table():
	df = pd.DataFrame()
	df["Lokalwahl"] = data_lokal.wähler/data_lokal.berechtigte
	df["Briefwahl"] = data_lokal.sperr_w/data_lokal.berechtigte
	df["Ungültig"] = data_lokal.ungültig/data_lokal.wähler
	for p in parties:
		df[p] = data_lokal[p] / data_lokal.wähler
	return df.corr().to_html(float_format = lambda i:format(i," .03f"))	


make_winner_map("winners.png")
make_map((briefwahl_per_w*data_lokal.sperr_w+data_lokal.wähler)/data_lokal.berechtigte, "beteiligung.png")
make_map(briefwahl_per_w*data_lokal.sperr_w/(briefwahl_per_w*data_lokal.sperr_w+data_lokal.wähler), "briefwahl.png")
make_map(np.log10(data_lokal.berechtigte / data_lokal.area), "area.png")
make_pca_parties("parteien_pca.svg")
make_pca_places("bezirke_pca.svg")
make_corr_matrix("korrelation.png")
for i in range(PCA_COMPONENTS_MAPS):
	make_map(dat_comps[i], f"pca{i+1}.png")

r=jinja_env.get_template("index.html").render(resulttable = get_result_table(),
										      briefwahl_success = briefwahl_per_w,
										      electionname = WAHL_NAME,
										      berechtigte = gesamt.berechtigte,
										      wähler = gesamt.wähler,
										      beteiligung = gesamt.wähler/gesamt.berechtigte,
										      lokalwähler = lokal_gesamt.wähler,
										      lokalwahl_anteil = lokal_gesamt.wähler/gesamt.wähler,
										      briefwähler = brief_gesamt.wähler,
										      briefwahl_anteil = brief_gesamt.wähler/gesamt.wähler,
										      sperrw = lokal_gesamt.sperr_w,
										      corrtable = get_corr_table())
with open(os.path.join(OUTDIR, "index.html"),"w") as f:
	f.write(r)
	
r=jinja_env.get_template("pca.html").render(pcatable=get_pca_table(),
											num_pca_maps = PCA_COMPONENTS_MAPS)
with open(os.path.join(OUTDIR, "pca.html"),"w") as f:
	f.write(r)

os.mkdir(os.path.join(OUTDIR, "parteien"))
r = jinja_env.get_template("parteien.html").render(parties = parties)
with open(os.path.join(OUTDIR, "parteien.html"),"w") as f:
	f.write(r)
for p in parties:
	make_map(data_lokal[p] / data_lokal.wähler, f"parteien/{p}.png")

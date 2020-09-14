import pandas as pd
import geopandas
from matplotlib import pyplot as plt
import matplotlib
import contextily as ctx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import tabulate
import requests
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

OUTDIR = "out/"
parties = ["CDU","SPD","Grüne","FDP","Linke","BBB","AfD","Piraten","BIG","PARTEI","StephanPost","Volt"]
columns = ["datum","wahl","ags","gebiet_nr","gebiet_name","max_schnellmeldungen","schnellmeldungen","nicht_sperr_w","sperr_w","nicht_verzeichnis","berechtigte","wähler","wähler_wahlschein","ungültig","gültig"]+parties
party_colors = {"Grüne":"green","SPD":"red","CDU":"black","BBB":"blue","BIG":"brown"} # only needed for winning parties in each district. matplotlib colors
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
	os.system(f"rm -r {OUTDIR}")
os.mkdir(OUTDIR)



fn_geodata = URL_GEODATA.split("/")[-1]

if not os.path.exists(fn_geodata):
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


def make_map(data, outfn):
	plt.figure(dpi = PLOT_DPI, frameon = False)
	ax = data_lokal.plot(data, alpha=0.5, legend=True, figsize=(12,12))
	ax.set_axis_off()
	ax.set_aspect('equal')
	ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom = 13)
	plt.box(False)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTDIR, outfn))
	plt.close()
	



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
		s = (["(nicht)","(brief)","(ungültig)"]+parties)[i]
		l = [s]
		l += [format(k[i], ".03f") for k in comps[:PCA_COMPONENTS_TABLE]]
		table.append(l)
	headers=["Partei"]+[f"Komp. {i+1}" for i in range(PCA_COMPONENTS_TABLE)]
	return tabulate.tabulate(table, headers=headers, tablefmt="html")
	

def get_result_table():
	table = []
	brief_gesamt=data[data.berechtigte==0].sum()
	lokal_gesamt=data[data.berechtigte!=0].sum()
	gesamt=data.sum()
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
	for i,p in enumerate(["(nicht)","(brief)","(ungültig)"]+parties):
		x=comps[0][i]
		y=comps[1][i]
		X.append(x)
		Y.append(y)
		plt.annotate(p, xy=(x,y))
	plt.plot(X,Y,".")
	plt.xlabel("Komponente 1")
	plt.ylabel("Komponente 2")
	plt.title("PCA Parteien")
	plt.savefig(os.path.join(OUTDIR, outfn))
	plt.close()
	
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
	plt.close()


	
make_map((briefwahl_per_w*data_lokal.sperr_w+data_lokal.wähler)/data_lokal.berechtigte, "beteiligung.png")
make_map(np.log10(data_lokal.berechtigte / data_lokal.area), "area.png")
make_pca_parties("parteien_pca.png")
make_pca_places("bezirke_pca.png")
for i in range(PCA_COMPONENTS_MAPS):
	make_map(dat_comps[i], f"pca{i+1}.png")

r=jinja_env.get_template("index.html").render(resulttable = get_result_table(),
										      pcatable = get_pca_table(),
										      briefwahl_success = briefwahl_per_w,
										      num_pca_maps = PCA_COMPONENTS_MAPS)
with open(os.path.join(OUTDIR, "index.html"),"w") as f:
	f.write(r)


os.mkdir(os.path.join(OUTDIR, "parteien"))
r = jinja_env.get_template("parteien.html").render(parties = parties)
with open(os.path.join(OUTDIR, "parteien", "index.html"),"w") as f:
	f.write(r)
for p in parties:
	make_map(data_lokal[p] / data_lokal.wähler, f"parteien/{p}.png")

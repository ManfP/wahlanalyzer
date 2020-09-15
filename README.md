# wahlanalyzer
Einfaches Skript um Wahldaten auf Karten zu visualisieren.
Gebaut für die Kommunalwahl 2020 in Bonn.
Benötigt werden Ergebnisse auf Stimmbezirkebene als CSV (Format von votemanager.de) sowie Geodaten der Stimmbezirke als GeoJSON.

Generiert wird eine einfache HTML-Übersicht. Diese enthält:
* Kurze Ergebnisübersicht
* Karte mit den stärksten Kräften je Stimmbezirk
* Karte der Wahlbeteiligung
* Karte des Briefwahlanteils
* Karte mit Bevölkerungsdichte (anhand der Anzahl Wahlberechtigter)
* Korrelationen unter den Parteien bezüglich ihrer Stimmbezirksergebnisse
* Karten mit dem Parteiabschneiden pro Stimmbezirk
* Eine Hauptkomponentenanalyse, um die lokalen Unterschiede der Stimmbezirke in weniger Dimensionen darzustellen

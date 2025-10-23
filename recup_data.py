import mercantile
import requests
import os
import mapbox_vector_tile
from vt2geojson.tools import vt_bytes_to_geojson

# — CONFIGURATION à adapter —
access_token = "MLY|31473198838990767|b3bc430b2e0c5825485a10081e047d10"
# bbox (west, south, east, north) en lon/lat
west, south, east, north = ( -0.581768,47.456627,-0.513722,47.486871 )  # exemple pour Angers, à adapter -0.606102,47.423024,-0.464653,47.509595
# Dossier de sortie
output_dir = "mapillary_images"

# Couche de tuile Mapillary pour couverture images
tile_coverage = "mly1_public"
tile_layer = "mly1_public_image"

# Crée dossier de sortie
os.makedirs(output_dir, exist_ok=True)

# Choix d’un zoom de tuile (ex: 14 ou 15 selon densité). Plus zoom élevé = plus de tuile, mais résolution fine.
zoom = 17

# Étape 1 : générer les tuiles couvrant le bbox
tiles = list(mercantile.tiles(west, south, east, north, zoom))

print(f"[INFO] Nombre de tuiles à traiter : {len(tiles)}")

# Pour chaque tuile, récupérer les features (images) dans cette tuile
image_keys = set()
for tile in tiles:
    z, x, y = tile.z, tile.x, tile.y
    tile_url = (f"https://tiles.mapillary.com/maps/vtp/{tile_coverage}/2/{z}/{x}/{y}"
                f"?access_token={access_token}")
    resp = requests.get(tile_url)
    if resp.status_code != 200:
        print(f"[WARN] tuile {z}/{x}/{y} retourné {resp.status_code}")
        continue
    decoded = mapbox_vector_tile.decode(resp.content)
    print(decoded.keys())
    geojson = vt_bytes_to_geojson(resp.content, x, y, z, layer=tile_layer)
    for feat in geojson["features"]:
        # On vérifie si la coordonnée est à l’intérieur du bbox
        lon, lat = feat["geometry"]["coordinates"]
        if (lon >= west and lon <= east and lat >= south and lat <= north):
            props = feat["properties"]
            image_keys.add(props["key"])

print(f"[INFO] Nombre d’images candidates : {len(image_keys)}")

# Étape 2 : pour chaque image, requête Graph pour l’URL et téléchargement
for key in image_keys:
    # demander URL + géométrie si besoin
    graph_url = (f"https://graph.mapillary.com/{key}"
                 f"?access_token={access_token}&fields=thumb_2048_url,geometry,computed_compass_angle")
    r = requests.get(graph_url)
    if r.status_code != 200:
        print(f"[WARN] image {key} non trouvée ({r.status_code})")
        continue
    data = r.json()
    if "thumb_2048_url" not in data:
        print(f"[WARN] pas d’url image pour key {key}")
        continue
    img_url = data["thumb_2048_url"]
    # télécharger image
    img_resp = requests.get(img_url, stream=True)
    if img_resp.status_code == 200:
        out_path = os.path.join(output_dir, f"{key}.jpg")
        with open(out_path, "wb") as f:
            for chunk in img_resp.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"[WARN] téléchargement échoué pour {key} ({img_resp.status_code})")
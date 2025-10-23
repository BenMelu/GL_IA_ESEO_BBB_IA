import requests
import os
import time

# ======== CONFIGURATION ========
ACCESS_TOKEN = "MLY|31473198838990767|b3bc430b2e0c5825485a10081e047d10"  # Mets ici ton token complet
BBOX = (-0.606102, 47.423024, -0.464653, 47.509595)  # Angers
STEP = 0.01  # Taille de découpe (en degrés) – plus petit = plus précis, mais plus lent
OUTPUT_DIR = "mapillary_images"
LIMIT = 1000  # Nombre max d’images par requête
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def bbox_split(west, south, east, north, step):
    """Génère des sous-bbox à partir d’un grand rectangle."""
    boxes = []
    lon = west
    while lon < east:
        lat = south
        next_lon = min(lon + step, east)
        while lat < north:
            next_lat = min(lat + step, north)
            boxes.append((lon, lat, next_lon, next_lat))
            lat = next_lat
        lon = next_lon
    return boxes


def fetch_images_for_bbox(bbox):
    """Récupère les images dans une sous-zone donnée."""
    west, south, east, north = bbox
    bbox_str = f"{west},{south},{east},{north}"
    url = (
        "https://graph.mapillary.com/images"
        f"?access_token={ACCESS_TOKEN}"
        f"&fields=id,geometry,thumb_2048_url"
        f"&bbox={bbox_str}"
        f"&limit={LIMIT}"
    )
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json().get("data", [])
        else:
            print(f"[WARN] bbox {bbox_str} → {r.status_code}")
            return []
    except Exception as e:
        print(f"[ERROR] Exception pour bbox {bbox_str}: {e}")
        return []


def download_image(img):
    """Télécharge une image Mapillary donnée."""
    url = img.get("thumb_2048_url")
    if not url:
        return
    img_id = img["id"]
    out_path = os.path.join(OUTPUT_DIR, f"{img_id}.jpg")
    if os.path.exists(out_path):
        return
    try:
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            print(f"Téléchargé : {img_id}")
        else:
            print(f"[WARN] Échec {img_id} ({resp.status_code})")
    except Exception as e:
        print(f"[ERROR] Téléchargement {img_id}: {e}")


def main():
    west, south, east, north = BBOX
    subboxes = bbox_split(west, south, east, north, STEP)
    print(f"[INFO] {len(subboxes)} sous-zones à explorer")

    total_images = 0
    for i, box in enumerate(subboxes, 1):
        print(f"[INFO] ({i}/{len(subboxes)}) Zone {box}")
        imgs = fetch_images_for_bbox(box)
        total_images += len(imgs)
        for img in imgs:
            download_image(img)
        # Pause légère pour éviter les limites de rate API
        time.sleep(0.5)

    print(f"[INFO] Total d’images traitées : {total_images}")


if __name__ == "__main__":
    main()
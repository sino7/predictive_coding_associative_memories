MODEL_DIR=saved_models
ATTN_MODEL_FILE=saved_models/latest_net_Attn.pth
CVAE_MODEL_FILE=saved_models/latest_net_CVAE.pth

mkdir $MODEL_DIR

g_download() {
	gURL="$1"
	dst="$2"

	# match more than 26 word characters
	ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')

	ggURL='https://drive.google.com/uc?export=download'

	curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null
	getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

	echo -e "Downloading from "$gURL"...\n"
	(cd $(dirname "$dst") && curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "$dst")
}

g_download "https://drive.google.com/open?id=1S4p3WL7QB67C9h91--B1G4XNIp2VrauE" "$ATTN_MODEL_FILE"
g_download "https://drive.google.com/open?id=1fA8ODaXhQE1rySH_L8PRVeMVjFG0pbQk" "$CVAE_MODEL_FILE"

shasums="$(mktemp)"
cat > "$shasums" <<EOF
0d2aeaac7dcc19181aeb84555b26ce51fe8aac2d  $ATTN_MODEL_FILE
3a4f2cc31147f4a12b7b52623574e9a1ac8ed056  $CVAE_MODEL_FILE
EOF
sha1sum -c "$shasums"
rm "$shasums"


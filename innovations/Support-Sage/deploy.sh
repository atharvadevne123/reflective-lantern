#!/usr/bin/env bash
# Deploy Support-Sage as a standalone GitHub repository.
# Usage: GH_PAT=<your-token> bash deploy.sh
set -e
GH_PAT="${GH_PAT:?Set GH_PAT environment variable}"
PROJECT="Support-Sage"
OWNER="atharvadevne123"

echo "Creating GitHub repo $OWNER/$PROJECT ..."
curl -sf -X POST \
  -H "Authorization: Bearer $GH_PAT" \
  -H "Content-Type: application/json" \
  "https://api.github.com/user/repos" \
  -d "{\"name\":\"$PROJECT\",\"description\":\"IT helpdesk ticket triage API predicting priority, category, and resolution time using XGBoost-LightGBM ensemble, NLP features, FAISS-based similar ticket retrieval, and KS-drift monitoring.\",\"public\":true}" \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print('Created:', r.get('html_url', r))"

echo "Pushing code ..."
git remote add origin "https://x-access-token:${GH_PAT}@github.com/${OWNER}/${PROJECT}"
git push -u origin main

echo "Done! Repo: https://github.com/$OWNER/$PROJECT"

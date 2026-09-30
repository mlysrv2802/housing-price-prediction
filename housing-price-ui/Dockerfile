# Dockerfile (for housing-price-ui)
# ------------------------------------
# MULTI-STAGE build -- two separate stages in one file:
#   Stage 1 ("build"): uses Node.js to compile the Angular app
#   Stage 2: uses nginx (a lightweight web server) to SERVE the compiled output
#
# Why two stages instead of one: Stage 1's image contains Node.js, npm,
# all your node_modules, and your TypeScript source -- none of which are
# needed to actually SERVE the app afterward. Stage 2 copies ONLY the
# compiled dist/ output into a fresh, minimal nginx image, discarding
# everything else. Final image ends up dramatically smaller than if we
# shipped Node.js + source code just to serve static files.

# --- Stage 1: Build -----------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Copy package files FIRST (same layer-caching trick as the Python
# Dockerfile) -- npm install only re-runs if package.json actually changed.
COPY package.json package-lock.json ./
RUN npm install

# Now copy the rest of the source code and build.
COPY . .
RUN npm run build

# --- Stage 2: Serve -------------------------------------------------------
FROM nginx:alpine

# Copy ONLY the compiled output from Stage 1 into nginx's default folder
# for static files. Note the exact path -- must match your actual
# dist/<project-name>/browser/ structure (Angular 17+).
COPY --from=build /app/dist/housing-price-ui/browser /usr/share/nginx/html

EXPOSE 80

# nginx's own default image already starts nginx automatically -- no CMD
# needed here, unlike the Python Dockerfile where we had to specify uvicorn.

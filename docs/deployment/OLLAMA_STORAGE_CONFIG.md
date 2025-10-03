# 🗄️ Ollama Storage Configuration - Using Z Drive

**Problem**: Ollama models are large (274 MB - 10+ GB) and can fill up C: drive quickly.

**Solution**: Configure Docker Compose to store Ollama models on Z: drive with plenty of space.

---

## ✅ Configuration Applied

### Docker Compose Changes

**File**: `docker-compose.yml`

**Before**:
```yaml
ollama:
  image: ollama/ollama:latest
  volumes:
    - ollama_data:/root/.ollama  # Docker managed volume on C: drive
```

**After**:
```yaml
ollama:
  image: ollama/ollama:latest
  volumes:
    - Z:/Ollama:/root/.ollama  # Direct mount to Z: drive
```

**Volume Definition Removed**:
```yaml
volumes:
  # ollama_data removed - using Z:/Ollama direct mount instead
  postgres_data:
    driver: local
  neo4j_data:
    driver: local
```

---

## 📦 What Gets Stored

The `Z:/Ollama` directory will contain:

```
Z:/Ollama/
├── models/
│   ├── manifests/
│   │   └── registry.ollama.ai/
│   │       └── library/
│   │           └── nomic-embed-text/
│   │               └── latest
│   └── blobs/
│       ├── sha256-970aa74c0a90...  (274 MB - model weights)
│       ├── sha256-c71d239df917...  (11 KB - config)
│       └── ...
└── .ollama/
    └── id_ed25519  (identity file)
```

**Typical Sizes**:
- **nomic-embed-text**: 274 MB
- **llama2-7b**: ~3.8 GB
- **llama2-13b**: ~7.4 GB
- **llama3-70b**: ~40 GB
- **mixtral-8x7b**: ~26 GB

---

## 🔄 Migration Steps (Already Completed)

### 1. Stop Ollama Container
```powershell
docker compose stop ollama
```

### 2. Remove Old Container
```powershell
docker compose rm -f ollama
```

### 3. Update docker-compose.yml
- Changed volume mount to `Z:/Ollama:/root/.ollama`
- Removed `ollama_data` from volumes section

### 4. Create Z: Drive Directory
```powershell
# Windows automatically creates directory when container starts
# Or create manually:
New-Item -ItemType Directory -Path "Z:\Ollama" -Force
```

### 5. Start New Container
```powershell
docker compose up -d ollama
```

### 6. Pull Models to New Location
```powershell
# Pull embedding model (274 MB)
docker compose exec ollama ollama pull nomic-embed-text

# Pull other models as needed
docker compose exec ollama ollama pull llama2
```

---

## 📊 Storage Benefits

### Before (C: Drive)
```
C:\ProgramData\Docker\volumes\agent_mem_ollama_data\
├── nomic-embed-text: 274 MB
├── llama2-7b: 3.8 GB
├── mixtral: 26 GB
└── Total: ~30 GB on C: drive ❌
```

### After (Z: Drive)
```
Z:\Ollama\
├── nomic-embed-text: 274 MB
├── llama2-7b: 3.8 GB
├── mixtral: 26 GB
└── Total: ~30 GB on Z: drive ✅
```

**Space Saved on C: Drive**: Up to 30+ GB depending on models

---

## 🔍 Verification

### Check Storage Location
```powershell
# List Z: drive contents
Get-ChildItem Z:\Ollama -Recurse

# Check disk usage
Get-ChildItem Z:\Ollama -Recurse | Measure-Object -Property Length -Sum | 
  Select-Object @{Name="Size (MB)";Expression={[math]::Round($_.Sum/1MB,2)}}
```

### Verify Models Are Accessible
```powershell
# List available models
docker compose exec ollama ollama list

# Expected output:
# NAME                       ID              SIZE      MODIFIED
# nomic-embed-text:latest    0a109f422b47    274 MB    ...
```

### Test Model Inference
```powershell
# Test embedding generation
docker compose exec ollama ollama run nomic-embed-text "test embedding"
```

---

## 🔧 Managing Models

### List All Models
```powershell
docker compose exec ollama ollama list
```

### Pull New Models
```powershell
# Embedding models
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull mxbai-embed-large

# Language models
docker compose exec ollama ollama pull llama2
docker compose exec ollama ollama pull llama3
docker compose exec ollama ollama pull mistral
docker compose exec ollama ollama pull mixtral
```

### Remove Models to Free Space
```powershell
# Remove a model
docker compose exec ollama ollama rm llama2

# Models are stored in Z:\Ollama\models\blobs\
# Remove manually if needed:
Remove-Item Z:\Ollama\models\* -Recurse -Force
```

### Check Model Details
```powershell
# Show model information
docker compose exec ollama ollama show nomic-embed-text
```

---

## 📁 Directory Structure Explained

```
Z:\Ollama/
├── models/
│   ├── manifests/              # Model metadata and configs
│   │   └── registry.ollama.ai/
│   │       └── library/
│   │           ├── nomic-embed-text/
│   │           ├── llama2/
│   │           └── ...
│   └── blobs/                  # Actual model files (large)
│       ├── sha256-XXX          # Model weights
│       ├── sha256-YYY          # Configurations
│       └── ...
└── .ollama/
    └── id_ed25519              # Ollama identity file
```

**Key Files**:
- **manifests/**: Model metadata (~1-10 KB each)
- **blobs/**: Model weights (100 MB - 40 GB each)
- **id_ed25519**: Ollama instance identity

---

## 🔄 Switching Storage Location

### To Change to Another Drive (e.g., D: drive)

1. **Update docker-compose.yml**:
```yaml
ollama:
  volumes:
    - D:/Ollama:/root/.ollama  # Change Z: to D:
```

2. **Recreate container**:
```powershell
docker compose down ollama
docker compose up -d ollama
```

3. **Copy existing models** (optional):
```powershell
# Copy from Z: to D:
Copy-Item -Path Z:\Ollama\* -Destination D:\Ollama\ -Recurse -Force
```

4. **Pull models** (if not copying):
```powershell
docker compose exec ollama ollama pull nomic-embed-text
```

---

## 🔒 Backup and Restore

### Backup Models
```powershell
# Backup entire Ollama directory
$backupDate = Get-Date -Format "yyyy-MM-dd"
Compress-Archive -Path Z:\Ollama\* `
  -DestinationPath "Z:\Backups\Ollama_$backupDate.zip" `
  -CompressionLevel Optimal
```

### Restore Models
```powershell
# Stop Ollama
docker compose stop ollama

# Restore from backup
Expand-Archive -Path Z:\Backups\Ollama_2025-10-02.zip `
  -DestinationPath Z:\Ollama\ -Force

# Start Ollama
docker compose up -d ollama
```

---

## 🎯 Best Practices

### 1. Regular Cleanup
```powershell
# Remove unused models monthly
docker compose exec ollama ollama list
docker compose exec ollama ollama rm <unused-model>
```

### 2. Monitor Disk Usage
```powershell
# Check Z: drive space
Get-PSDrive Z | Select-Object Name, 
  @{Name="Used (GB)";Expression={[math]::Round($_.Used/1GB,2)}},
  @{Name="Free (GB)";Expression={[math]::Round($_.Free/1GB,2)}}
```

### 3. Model Selection
- **For Embeddings**: Use `nomic-embed-text` (274 MB) - fast and efficient
- **For Chat**: Start with `llama2` (3.8 GB) before trying larger models
- **For Speed**: Prefer quantized models (Q4, Q5) over full precision

### 4. Periodic Backups
```powershell
# Create monthly backups
$schedule = New-ScheduledTaskAction -Execute 'PowerShell.exe' `
  -Argument '-File "C:\Scripts\backup-ollama.ps1"'
```

---

## 🐛 Troubleshooting

### Issue: "Cannot find Z: drive"
**Solution**: Ensure Z: drive is mounted and accessible
```powershell
Get-PSDrive -PSProvider FileSystem
# If missing, mount network drive or external drive as Z:
```

### Issue: "Permission denied" when mounting
**Solution**: Run Docker Desktop as Administrator or grant permissions
```powershell
# Grant full control to Docker
icacls Z:\Ollama /grant Everyone:F /T
```

### Issue: Models not appearing after migration
**Solution**: Pull models again or copy from old location
```powershell
# Re-pull model
docker compose exec ollama ollama pull nomic-embed-text

# Or copy from old Docker volume
docker cp old-container:/root/.ollama/models Z:\Ollama\
```

### Issue: "Disk space insufficient" on Z:
**Solution**: Clean up old models or use larger drive
```powershell
# Check what's using space
Get-ChildItem Z:\Ollama\models\blobs -Recurse | 
  Sort-Object Length -Descending | 
  Select-Object Name, @{Name="Size (MB)";Expression={[math]::Round($_.Length/1MB,2)}} -First 10
```

---

## 📊 Current Configuration

**Storage Location**: `Z:\Ollama`  
**Docker Mount**: `Z:/Ollama:/root/.ollama`  
**Current Models**: 
- `nomic-embed-text:latest` - 274 MB ✅

**Space Available on Z:**: Check with:
```powershell
Get-PSDrive Z | Select-Object Free
```

---

## ✨ Benefits of This Configuration

1. ✅ **Saves C: Drive Space**: Models stored on separate drive
2. ✅ **Easy Backup**: Direct access to `Z:\Ollama` folder
3. ✅ **Better Performance**: Dedicated drive for AI models
4. ✅ **Portable**: Can move Z: drive to another machine
5. ✅ **No Docker Volume Hassle**: Direct filesystem access
6. ✅ **Easy Monitoring**: Can see space usage in Windows Explorer

---

## 🔗 Related Commands

```powershell
# Check Ollama service
docker compose ps ollama

# View Ollama logs
docker compose logs ollama

# Restart Ollama
docker compose restart ollama

# Access Ollama API
curl http://localhost:11434/api/tags

# Execute command in container
docker compose exec ollama sh
```

---

**Status**: ✅ Configured and Active  
**Storage Location**: Z:\Ollama  
**Space Saved**: Up to 30+ GB on C: drive  
**Last Updated**: October 2, 2025

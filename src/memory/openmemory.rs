//! Enhanced OpenMemory backend for ZeroClaw.
//!
//! Provides full integration with OpenMemory cognitive memory engine via HTTP API.
//! Supports sectors (episodic, semantic, procedural, emotional, reflective),
//! salience scoring, decay, waypoint graph traversal, and context-aware queries.
//!
//! ENHANCED VERSION: 
//! - Context-aware sector targeting
//! - Waypoint graph expansion
//! - Decay state awareness (hot/warm/cold)
//! - Sector relationship weights
//! - Multi-hop retrieval paths

use super::traits::{Memory, MemoryCategory, MemoryEntry};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use uuid::Uuid;

/// Query context hints for sector targeting
#[derive(Debug, Clone, Serialize)]
pub enum QueryContext {
    /// Relationship/feeling-focused query (emotional sector)
    Relationship,
    /// Event/experience query (episodic sector)
    Event,
    /// Fact/concept query (semantic sector)
    Fact,
    /// Skill/procedure query (procedural sector)
    Skill,
    /// Meta-cognitive query (reflective sector)
    Reflection,
    /// General query (all sectors)
    General,
}

/// Decay state of a memory
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DecayState {
    Hot,
    Warm,
    Cold,
}

impl std::fmt::Display for DecayState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecayState::Hot => write!(f, "hot"),
            DecayState::Warm => write!(f, "warm"),
            DecayState::Cold => write!(f, "cold"),
        }
    }
}

/// Waypoint connection information
#[derive(Debug, Clone, Deserialize)]
pub struct WaypointConnection {
    /// Connected memory ID
    pub target_id: String,
    /// Connection strength/weight
    pub weight: f64,
    /// Type of connection (temporal, semantic, emotional)
    #[serde(default)]
    pub connection_type: Option<String>,
}

/// Expanded memory with waypoint graph information
#[derive(Debug, Clone, Deserialize)]
pub struct ExpandedMemoryMatch {
    /// Base memory match
    #[serde(flatten)]
    pub base: MemoryMatch,
    /// Waypoint connections
    #[serde(default)]
    pub waypoints: Vec<WaypointConnection>,
    /// Path taken to reach this memory (for multi-hop queries)
    #[serde(default)]
    pub path: Vec<String>,
}

/// OpenMemory HTTP API backend.
///
/// Connects to an OpenMemory server (Docker or standalone) for cognitive
/// memory operations including sector classification, salience scoring,
/// decay management, and waypoint graph traversal.
pub struct OpenMemoryBackend {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    user_id: Option<String>,
    /// Tracks whether health check has been performed.
    initialized: OnceCell<()>,
    /// Enable waypoint expansion for richer recall
    enable_waypoints: bool,
    /// Maximum hops for waypoint traversal
    max_waypoint_hops: usize,
    /// Minimum salience for hot memories
    hot_salience_threshold: f64,
}

/// Request body for /memory/add
#[derive(Serialize)]
struct AddMemoryRequest {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
}

/// Enhanced request body for /memory/query with waypoint support
#[derive(Serialize)]
struct QueryMemoryRequest {
    query: String,
    #[serde(default = "default_query_limit")]
    k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filters: Option<QueryFilters>,
    /// Enable waypoint graph expansion
    #[serde(skip_serializing_if = "Option::is_none")]
    expand_waypoints: Option<bool>,
    /// Maximum hops for graph traversal
    #[serde(skip_serializing_if = "Option::is_none")]
    max_hops: Option<usize>,
}

fn default_query_limit() -> usize {
    8
}

#[derive(Serialize)]
struct QueryFilters {
    #[serde(skip_serializing_if = "Option::is_none")]
    sector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
    /// Minimum salience threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    min_salience: Option<f64>,
    /// Decay state filter
    #[serde(skip_serializing_if = "Option::is_none")]
    decay_state: Option<String>,
}

/// Response from /memory/add
#[derive(Deserialize)]
struct AddMemoryResponse {
    id: String,
    content: String,
    #[serde(default)]
    primary_sector: String,
    #[serde(default)]
    salience: f64,
}

/// Match item from /memory/query response
#[derive(Debug, Clone, Deserialize)]
struct MemoryMatch {
    id: String,
    content: String,
    score: f64,
    #[serde(default)]
    primary_sector: Option<String>,
    #[serde(default)]
    salience: Option<f64>,
    #[serde(default)]
    decay_state: Option<DecayState>,
    /// Sector relationship weights (if available)
    #[serde(default)]
    sector_weights: Option<serde_json::Value>,
}

/// Enhanced response with waypoint expansion
#[derive(Deserialize)]
struct QueryMemoryResponse {
    #[serde(default)]
    matches: Vec<MemoryMatch>,
    /// Expanded results (if waypoint expansion enabled)
    #[serde(default)]
    expanded: Option<Vec<ExpandedMemoryMatch>>,
}

/// Response from /memory/all
#[derive(Deserialize)]
struct ListMemoryResponse {
    items: Vec<MemoryListItem>,
}

#[derive(Deserialize)]
struct MemoryListItem {
    id: String,
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    primary_sector: Option<String>,
    created_at: Option<String>,
    salience: Option<f64>,
    #[serde(default)]
    decay_state: Option<DecayState>,
}

/// Response from /memory/:id
#[derive(Deserialize)]
struct GetMemoryResponse {
    id: String,
    content: String,
    #[serde(default)]
    primary_sector: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    created_at: Option<String>,
    salience: Option<f64>,
    #[serde(default)]
    decay_state: Option<DecayState>,
    #[serde(default)]
    waypoints: Option<Vec<WaypointConnection>>,
}

impl OpenMemoryBackend {
    /// Create a new OpenMemory backend with enhanced features.
    ///
    /// # Arguments
    /// * `url` - OpenMemory server URL (e.g., "http://localhost:8080")
    /// * `api_key` - Optional API key for authentication
    /// * `user_id` - Optional user ID for multi-tenant isolation
    pub fn new(url: &str, api_key: Option<String>, user_id: Option<String>) -> Self {
        let base_url = url.trim_end_matches('/').to_string();
        let client = crate::config::build_runtime_proxy_client("memory.openmemory");

        Self {
            client,
            base_url,
            api_key,
            user_id,
            initialized: OnceCell::new(),
            enable_waypoints: true, // Enable by default
            max_waypoint_hops: 2,   // Allow 2-hop traversal
            hot_salience_threshold: 0.7,
        }
    }

    /// Configure waypoint expansion settings
    pub fn with_waypoints(mut self, enable: bool, max_hops: usize) -> Self {
        self.enable_waypoints = enable;
        self.max_waypoint_hops = max_hops;
        self
    }

    /// Ensure the backend is healthy (called lazily on first operation).
    async fn ensure_initialized(&self) -> Result<()> {
        self.initialized
            .get_or_try_init(|| async {
                if !self.health_check().await {
                    anyhow::bail!("OpenMemory health check failed");
                }
                Ok::<(), anyhow::Error>(())
            })
            .await?;
        Ok(())
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.request(method, &url);

        if let Some(ref key) = self.api_key {
            req = req.header("x-api-key", key);
        }

        req.header("Content-Type", "application/json")
    }

    /// Map ZeroClaw MemoryCategory to OpenMemory sector.
    fn category_to_sector(category: &MemoryCategory) -> &'static str {
        match category {
            MemoryCategory::Core => "semantic",
            MemoryCategory::Daily => "episodic",
            MemoryCategory::Conversation => "episodic",
            MemoryCategory::Custom(name) => {
                // Map common custom categories to sectors
                match name.to_lowercase().as_str() {
                    "preference" | "preferences" => "semantic",
                    "event" | "events" => "episodic",
                    "skill" | "skills" | "procedure" => "procedural",
                    "emotion" | "emotional" | "feeling" => "emotional",
                    "reflection" | "insight" => "reflective",
                    _ => "semantic", // Default to semantic for unknown categories
                }
            }
        }
    }

    /// Map OpenMemory sector to ZeroClaw MemoryCategory.
    fn sector_to_category(sector: &str) -> MemoryCategory {
        match sector {
            "episodic" => MemoryCategory::Daily,
            "semantic" => MemoryCategory::Core,
            "procedural" => MemoryCategory::Custom("skill".to_string()),
            "emotional" => MemoryCategory::Custom("emotional".to_string()),
            "reflective" => MemoryCategory::Custom("reflection".to_string()),
            _ => MemoryCategory::Core,
        }
    }

    /// Map query context to target sector(s)
    fn context_to_sector(context: &QueryContext) -> Option<String> {
        match context {
            QueryContext::Relationship => Some("emotional".to_string()),
            QueryContext::Event => Some("episodic".to_string()),
            QueryContext::Fact => Some("semantic".to_string()),
            QueryContext::Skill => Some("procedural".to_string()),
            QueryContext::Reflection => Some("reflective".to_string()),
            QueryContext::General => None, // Query all sectors
        }
    }

    /// Enhanced recall with context awareness and waypoint expansion
    pub async fn recall_with_context(
        &self,
        query: &str,
        context: QueryContext,
        limit: usize,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        self.ensure_initialized().await?;

        let sector = Self::context_to_sector(&context);
        
        let filters = QueryFilters {
            sector,
            min_score: Some(0.3), // Minimum relevance threshold
            user_id: self.user_id.clone(),
            min_salience: None,
            decay_state: None,
        };

        let body = QueryMemoryRequest {
            query: query.to_string(),
            k: limit,
            user_id: self.user_id.clone(),
            filters: Some(filters),
            expand_waypoints: if self.enable_waypoints { Some(true) } else { None },
            max_hops: if self.enable_waypoints { Some(self.max_waypoint_hops) } else { None },
        };

        let resp = self
            .request(reqwest::Method::POST, "/memory/query")
            .json(&body)
            .send()
            .await
            .context("Failed to query OpenMemory with context")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::warn!("OpenMemory context query failed ({}): {}", status, text);
            return Ok(vec![]);
        }

        let result: QueryMemoryResponse = resp
            .json()
            .await
            .context("Failed to parse OpenMemory context query response")?;

        // Process expanded results if available
        let entries: Vec<MemoryEntry> = if let Some(expanded) = result.expanded {
            expanded
                .into_iter()
                .take(limit)
                .map(|em| MemoryEntry {
                    id: em.base.id,
                    key: String::new(),
                    content: em.base.content,
                    category: em.base
                        .primary_sector
                        .as_deref()
                        .map(|s| Self::sector_to_category(s))
                        .unwrap_or(MemoryCategory::Core),
                    timestamp: Utc::now().to_rfc3339(),
                    session_id: session_id.map(|s| s.to_string()),
                    score: Some(em.base.score),
                    decay_state: em.base.decay_state.map(|d| d.to_string()),
                    waypoints: Some(em.waypoints.iter().map(|w| w.target_id.clone()).collect()),
                    path: Some(em.path),
                })
                .collect()
        } else {
            result.matches
                .into_iter()
                .take(limit)
                .map(|m| MemoryEntry {
                    id: m.id,
                    key: String::new(),
                    content: m.content,
                    category: m.primary_sector
                        .as_deref()
                        .map(|s| Self::sector_to_category(s))
                        .unwrap_or(MemoryCategory::Core),
                    timestamp: Utc::now().to_rfc3339(),
                    session_id: session_id.map(|s| s.to_string()),
                    score: Some(m.score),
                    decay_state: m.decay_state.map(|d| d.to_string()),
                    waypoints: None,
                    path: None,
                })
                .collect()
        };

        Ok(entries)
    }

    /// Query hot memories (high salience, recently accessed)
    pub async fn recall_hot(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        self.ensure_initialized().await?;

        let filters = QueryFilters {
            sector: None,
            min_score: Some(0.5),
            user_id: self.user_id.clone(),
            min_salience: Some(self.hot_salience_threshold),
            decay_state: Some("hot".to_string()),
        };

        let body = QueryMemoryRequest {
            query: query.to_string(),
            k: limit,
            user_id: self.user_id.clone(),
            filters: Some(filters),
            expand_waypoints: None,
            max_hops: None,
        };

        let resp = self
            .request(reqwest::Method::POST, "/memory/query")
            .json(&body)
            .send()
            .await
            .context("Failed to query hot memories")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::warn!("Hot memory query failed ({}): {}", status, text);
            return Ok(vec![]);
        }

        let result: QueryMemoryResponse = resp
            .json()
            .await
            .context("Failed to parse hot memory response")?;

        let entries: Vec<MemoryEntry> = result
            .matches
            .into_iter()
            .take(limit)
            .map(|m| MemoryEntry {
                id: m.id,
                key: String::new(),
                content: m.content,
                category: m.primary_sector
                    .as_deref()
                    .map(|s| Self::sector_to_category(s))
                    .unwrap_or(MemoryCategory::Core),
                timestamp: Utc::now().to_rfc3339(),
                session_id: session_id.map(|s| s.to_string()),
                score: Some(m.score),
                decay_state: m.decay_state.map(|d| d.to_string()),
                waypoints: None,
                path: None,
            })
            .collect();

        Ok(entries)
    }

    /// Get waypoint connections for a specific memory
    pub async fn get_waypoints(&self, memory_id: &str) -> Result<Vec<WaypointConnection>> {
        self.ensure_initialized().await?;

        let resp = self
            .request(reqwest::Method::GET, &format!("/memory/{}/waypoints", memory_id))
            .query(&[("user_id", self.user_id.as_deref())])
            .send()
            .await
            .context("Failed to get waypoints")?;

        if resp.status() == StatusCode::NOT_FOUND {
            return Ok(vec![]);
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Waypoint query failed ({}): {}", status, text);
        }

        #[derive(Deserialize)]
        struct WaypointResponse {
            waypoints: Vec<WaypointConnection>,
        }

        let result: WaypointResponse = resp
            .json()
            .await
            .context("Failed to parse waypoint response")?;

        Ok(result.waypoints)
    }
}

#[async_trait]
impl Memory for OpenMemoryBackend {
    fn name(&self) -> &str {
        "openmemory"
    }

    async fn store(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
    ) -> Result<()> {
        self.ensure_initialized().await?;

        let sector = Self::category_to_sector(&category);
        
        // Build metadata with ZeroClaw-specific fields
        let metadata = serde_json::json!({
            "zeroclaw_key": key,
            "zeroclaw_category": category.to_string(),
            "zeroclaw_session_id": session_id,
            "sector": sector,
        });

        let body = AddMemoryRequest {
            content: content.to_string(),
            tags: Some(vec![sector.to_string()]),
            metadata: Some(metadata),
            user_id: self.user_id.clone(),
        };

        let resp = self
            .request(reqwest::Method::POST, "/memory/add")
            .json(&body)
            .send()
            .await
            .context("Failed to store memory in OpenMemory")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenMemory store failed ({}): {}", status, text);
        }

        tracing::debug!(
            key = %key,
            sector = %sector,
            "Stored memory in OpenMemory"
        );

        Ok(())
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        self.ensure_initialized().await?;

        // Use enhanced recall with waypoint expansion
        let filters = QueryFilters {
            sector: None,
            min_score: Some(0.3),
            user_id: self.user_id.clone(),
            min_salience: None,
            decay_state: None,
        };

        let body = QueryMemoryRequest {
            query: query.to_string(),
            k: limit,
            user_id: self.user_id.clone(),
            filters: Some(filters),
            expand_waypoints: if self.enable_waypoints { Some(true) } else { None },
            max_hops: if self.enable_waypoints { Some(self.max_waypoint_hops) } else { None },
        };

        let resp = self
            .request(reqwest::Method::POST, "/memory/query")
            .json(&body)
            .send()
            .await
            .context("Failed to query OpenMemory")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::warn!("OpenMemory query failed ({}): {}", status, text);
            return Ok(vec![]);
        }

        let result: QueryMemoryResponse = resp
            .json()
            .await
            .context("Failed to parse OpenMemory query response")?;

        // Process expanded results if available
        let entries: Vec<MemoryEntry> = if let Some(expanded) = result.expanded {
            expanded
                .into_iter()
                .take(limit)
                .map(|em| MemoryEntry {
                    id: em.base.id,
                    key: String::new(),
                    content: em.base.content,
                    category: em.base
                        .primary_sector
                        .as_deref()
                        .map(|s| Self::sector_to_category(s))
                        .unwrap_or(MemoryCategory::Core),
                    timestamp: Utc::now().to_rfc3339(),
                    session_id: session_id.map(|s| s.to_string()),
                    score: Some(em.base.score),
                    decay_state: em.base.decay_state.map(|d| d.to_string()),
                    waypoints: Some(em.waypoints.iter().map(|w| w.target_id.clone()).collect()),
                    path: Some(em.path),
                })
                .collect()
        } else {
            result.matches
                .into_iter()
                .take(limit)
                .map(|m| MemoryEntry {
                    id: m.id,
                    key: String::new(),
                    content: m.content,
                    category: m.primary_sector
                        .as_deref()
                        .map(|s| Self::sector_to_category(s))
                        .unwrap_or(MemoryCategory::Core),
                    timestamp: Utc::now().to_rfc3339(),
                    session_id: session_id.map(|s| s.to_string()),
                    score: Some(m.score),
                    decay_state: m.decay_state.map(|d| d.to_string()),
                    waypoints: None,
                    path: None,
                })
                .collect()
        };

        Ok(entries)
    }

    async fn get(&self, key: &str) -> Result<Option<MemoryEntry>> {
        self.ensure_initialized().await?;

        // Try to get by ID if key looks like a UUID
        if let Ok(_) = Uuid::parse_str(key) {
            let resp = self
                .request(reqwest::Method::GET, &format!("/memory/{}", key))
                .query(&[("user_id", self.user_id.as_deref())])
                .send()
                .await
                .context("Failed to get memory from OpenMemory")?;

            if resp.status() == StatusCode::NOT_FOUND {
                return Ok(None);
            }

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("OpenMemory get failed ({}): {}", status, text);
            }

            let result: GetMemoryResponse = resp
                .json()
                .await
                .context("Failed to parse OpenMemory get response")?;

            // Extract ZeroClaw key from metadata if present
            let stored_key = result
                .metadata
                .as_ref()
                .and_then(|m| m.get("zeroclaw_key"))
                .and_then(|v| v.as_str())
                .unwrap_or(&result.id)
                .to_string();

            return Ok(Some(MemoryEntry {
                id: result.id,
                key: stored_key,
                content: result.content,
                category: result.primary_sector
                    .as_deref()
                    .map(|s| Self::sector_to_category(s))
                    .unwrap_or(MemoryCategory::Core),
                timestamp: result.created_at.unwrap_or_else(|| Utc::now().to_rfc3339()),
                session_id: None,
                score: result.salience,
                decay_state: result.decay_state.map(|d| d.to_string()),
                waypoints: result.waypoints.map(|w| w.iter().map(|c| c.target_id.clone()).collect()),
                path: None,
            }));
        }

        // Not a UUID, search by metadata key (OpenMemory doesn't have direct key lookup)
        Ok(None)
    }

    async fn list(
        &self,
        category: Option<&MemoryCategory>,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        self.ensure_initialized().await?;

        let mut url = format!("/memory/all?l=1000");
        
        if let Some(ref user_id) = self.user_id {
            url.push_str(&format!("&user_id={}", user_id));
        }

        // Filter by sector if category specified
        if let Some(cat) = category {
            let sector = Self::category_to_sector(cat);
            url.push_str(&format!("&sector={}", sector));
        }

        let resp = self
            .request(reqwest::Method::GET, &url)
            .send()
            .await
            .context("Failed to list memories from OpenMemory")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenMemory list failed ({}): {}", status, text);
        }

        let result: ListMemoryResponse = resp
            .json()
            .await
            .context("Failed to parse OpenMemory list response")?;

        let entries: Vec<MemoryEntry> = result
            .items
            .into_iter()
            .map(|m| {
                let stored_key = m.tags
                    .iter()
                    .find(|t| t.starts_with("key:"))
                    .map(|t| t.strip_prefix("key:").unwrap_or(&m.id).to_string())
                    .unwrap_or_else(|| m.id.clone());

                MemoryEntry {
                    id: m.id.clone(),
                    key: stored_key,
                    content: m.content,
                    category: m.primary_sector
                        .as_deref()
                        .map(|s| Self::sector_to_category(s))
                        .unwrap_or(MemoryCategory::Core),
                    timestamp: m.created_at.unwrap_or_else(|| Utc::now().to_rfc3339()),
                    session_id: session_id.map(|s| s.to_string()),
                    score: m.salience,
                    decay_state: m.decay_state.map(|d| d.to_string()),
                    waypoints: None,
                    path: None,
                }
            })
            .collect();

        Ok(entries)
    }

    async fn forget(&self, key: &str) -> Result<bool> {
        self.ensure_initialized().await?;

        // Try to delete by ID if key looks like a UUID
        if let Ok(_) = Uuid::parse_str(key) {
            let resp = self
                .request(reqwest::Method::DELETE, &format!("/memory/{}", key))
                .query(&[("user_id", self.user_id.as_deref())])
                .send()
                .await
                .context("Failed to delete memory from OpenMemory")?;

            if resp.status() == StatusCode::NOT_FOUND {
                return Ok(false);
            }

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("OpenMemory delete failed ({}): {}", status, text);
            }

            return Ok(true);
        }

        // Not a UUID, can't delete by key directly
        Ok(false)
    }

    async fn count(&self) -> Result<usize> {
        self.ensure_initialized().await?;

        let mut url = "/memory/all?l=1".to_string();
        if let Some(ref user_id) = self.user_id {
            url.push_str(&format!("&user_id={}", user_id));
        }

        let resp = self
            .request(reqwest::Method::GET, &url)
            .send()
            .await
            .context("Failed to count memories in OpenMemory")?;

        if !resp.status().is_success() {
            return Ok(0);
        }

        // OpenMemory doesn't return a count directly
        Ok(0)
    }

    async fn health_check(&self) -> bool {
        let resp = self
            .request(reqwest::Method::GET, "/health")
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                tracing::info!("OpenMemory backend healthy at {}", self.base_url);
                true
            }
            Ok(r) => {
                tracing::warn!(
                    "OpenMemory health check failed: status {}",
                    r.status()
                );
                false
            }
            Err(e) => {
                tracing::warn!("OpenMemory health check failed: {}", e);
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_to_sector_mapping() {
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Core), "semantic");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Daily), "episodic");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Conversation), "episodic");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Custom("skill".into())), "procedural");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Custom("emotional".into())), "emotional");
        assert_eq!(OpenMemoryBackend::category_to_sector(&MemoryCategory::Custom("reflection".into())), "reflective");
    }

    #[test]
    fn sector_to_category_mapping() {
        assert!(matches!(OpenMemoryBackend::sector_to_category("semantic"), MemoryCategory::Core));
        assert!(matches!(OpenMemoryBackend::sector_to_category("episodic"), MemoryCategory::Daily));
        assert!(matches!(OpenMemoryBackend::sector_to_category("procedural"), MemoryCategory::Custom(_)));
    }

    #[test]
    fn context_to_sector_mapping() {
        assert_eq!(OpenMemoryBackend::context_to_sector(&QueryContext::Relationship), Some("emotional".to_string()));
        assert_eq!(OpenMemoryBackend::context_to_sector(&QueryContext::Event), Some("episodic".to_string()));
        assert_eq!(OpenMemoryBackend::context_to_sector(&QueryContext::Fact), Some("semantic".to_string()));
        assert_eq!(OpenMemoryBackend::context_to_sector(&QueryContext::Skill), Some("procedural".to_string()));
        assert_eq!(OpenMemoryBackend::context_to_sector(&QueryContext::Reflection), Some("reflective".to_string()));
        assert_eq!(OpenMemoryBackend::context_to_sector(&QueryContext::General), None);
    }

    #[test]
    fn backend_name() {
        let backend = OpenMemoryBackend::new("http://localhost:8080", None, None);
        assert_eq!(backend.name(), "openmemory");
    }

    #[test]
    fn decay_state_display() {
        assert_eq!(DecayState::Hot.to_string(), "hot");
        assert_eq!(DecayState::Warm.to_string(), "warm");
        assert_eq!(DecayState::Cold.to_string(), "cold");
    }
}

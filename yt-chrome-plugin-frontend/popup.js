// popup.js

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyBNwBulP6e-jyxWCC8I5QK51CHheBVkJEw';
  const API_URL = 'http://127.0.0.1:8000';

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];

      outputDiv.innerHTML = `
        <div class="video-id-tag"><span>ID</span>${videoId}</div>
        <div class="loading-text">Fetching comments...</div>
        <div class="loading-line"></div>
      `;

      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        outputDiv.innerHTML += `<div class="loading-text" style="color:#ff4444">No comments found.</div>`;
        return;
      }

      outputDiv.innerHTML += `
        <div class="loading-text">Running sentiment model on ${comments.length} comments...</div>
        <div class="loading-line"></div>
      `;

      const predictions = await getSentimentPredictions(comments);

      if (predictions) {
        const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
        const sentimentData = [];
        const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);

        predictions.forEach((item) => {
          sentimentCounts[item.sentiment]++;
          sentimentData.push({ timestamp: item.timestamp, sentiment: parseInt(item.sentiment) });
        });

        const totalComments = comments.length;
        const uniqueCommenters = new Set(comments.map(c => c.authorId)).size;
        const totalWords = comments.reduce((sum, c) => sum + c.text.split(/\s+/).filter(w => w.length > 0).length, 0);
        const avgWordLength = (totalWords / totalComments).toFixed(1);
        const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
        const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(1);

        // Reset output and build clean UI
        outputDiv.innerHTML = `
          <div class="video-id-tag"><span>ID</span>${videoId}</div>
        `;

        // Metrics
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-header">
              <span class="section-label">Overview</span>
              <div class="section-line"></div>
            </div>
            <div class="metrics-grid">
              <div class="metric-card">
                <div class="metric-title">Total Comments</div>
                <div class="metric-value">${totalComments}</div>
              </div>
              <div class="metric-card">
                <div class="metric-title">Unique Commenters</div>
                <div class="metric-value">${uniqueCommenters}</div>
              </div>
              <div class="metric-card">
                <div class="metric-title">Avg Length</div>
                <div class="metric-value">${avgWordLength}<span class="metric-unit">wds</span></div>
              </div>
              <div class="metric-card">
                <div class="metric-title">Sentiment Score</div>
                <div class="metric-value">${normalizedSentimentScore}<span class="metric-unit">/10</span></div>
              </div>
            </div>
          </div>
        `;

        // Pie Chart
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-header">
              <span class="section-label">Sentiment Distribution</span>
              <div class="section-line"></div>
            </div>
            <div class="viz-box" id="chart-container"></div>
          </div>
        `;
        await fetchAndDisplayChart(sentimentCounts);

        // Trend Graph
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-header">
              <span class="section-label">Sentiment Trend</span>
              <div class="section-line"></div>
            </div>
            <div class="viz-box" id="trend-graph-container"></div>
          </div>
        `;
        await fetchAndDisplayTrendGraph(sentimentData);

        // Word Cloud
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-header">
              <span class="section-label">Word Cloud</span>
              <div class="section-line"></div>
            </div>
            <div class="viz-box" id="wordcloud-container"></div>
          </div>
        `;
        await fetchAndDisplayWordCloud(comments.map(c => c.text));

        // Top Comments
        const sentimentLabel = (s) => {
          if (s === "1") return `<span class="sentiment-badge pos">POSITIVE</span>`;
          if (s === "-1") return `<span class="sentiment-badge neg">NEGATIVE</span>`;
          return `<span class="sentiment-badge neu">NEUTRAL</span>`;
        };

        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-header">
              <span class="section-label">Top 25 Comments</span>
              <div class="section-line"></div>
            </div>
            <ul class="comment-list">
              ${predictions.slice(0, 25).map((item, i) => `
                <li class="comment-item">
                  <div class="comment-number">#${String(i + 1).padStart(2, '0')}</div>
                  <div class="comment-text">${item.comment}</div>
                  ${sentimentLabel(item.sentiment)}
                </li>
              `).join('')}
            </ul>
          </div>
        `;
      }

    } else {
      outputDiv.innerHTML = `
        <div class="error-state">
          <div class="error-icon">⚠</div>
          <div class="error-title">Not a YouTube Video</div>
          <div class="error-sub">Navigate to a YouTube watch page to analyze comments.</div>
        </div>
      `;
    }
  });

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&order=relevance&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await response.json();
        if (data.items) {
          data.items.forEach(item => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
            comments.push({ text: commentText, timestamp, authorId });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      outputDiv.innerHTML += `<div class="loading-text" style="color:#ff4444">Error fetching comments.</div>`;
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      const result = await response.json();
      if (response.ok) return result;
      throw new Error(result.error || 'Error fetching predictions');
    } catch (error) {
      console.error("Error fetching predictions:", error);
      outputDiv.innerHTML += `<div class="loading-text" style="color:#ff4444">Error fetching predictions.</div>`;
      return null;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      if (!response.ok) throw new Error('Failed to fetch chart');
      const blob = await response.blob();
      const img = document.createElement('img');
      img.src = URL.createObjectURL(blob);
      document.getElementById('chart-container').appendChild(img);
    } catch (e) { console.error(e); }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) throw new Error('Failed to fetch wordcloud');
      const blob = await response.blob();
      const img = document.createElement('img');
      img.src = URL.createObjectURL(blob);
      document.getElementById('wordcloud-container').appendChild(img);
    } catch (e) { console.error(e); }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      if (!response.ok) throw new Error('Failed to fetch trend graph');
      const blob = await response.blob();
      const img = document.createElement('img');
      img.src = URL.createObjectURL(blob);
      document.getElementById('trend-graph-container').appendChild(img);
    } catch (e) { console.error(e); }
  }
});
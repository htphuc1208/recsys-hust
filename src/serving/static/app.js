/**
 * RecSys HUST — Modern Recommendation UI Interactive Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentPersonas = [];
    let selectedPersona = null;
    let currentHistoryIds = [];
    let currentRecommendations = [];
    let activeCategoryFilter = 'all';

    // DOM Elements
    const healthPill = document.getElementById('system-health-pill');
    const healthText = document.getElementById('system-health-text');
    const personasGrid = document.getElementById('personas-grid');
    const userAvatar = document.getElementById('current-user-avatar');
    const userName = document.getElementById('current-user-name');
    const userIdBadge = document.getElementById('current-user-id-badge');
    const userRole = document.getElementById('current-user-role');
    const userDesc = document.getElementById('current-user-desc');
    const historyChips = document.getElementById('history-chips-container');
    const btnClearHistory = document.getElementById('btn-clear-history');
    const topkSelect = document.getElementById('topk-select');
    const filterSeenToggle = document.getElementById('filter-seen-toggle');
    const btnRecommend = document.getElementById('btn-run-recommendation');
    const newsGrid = document.getElementById('news-grid');
    const filterPills = document.querySelectorAll('.filter-pill');
    
    // HUD Elements
    const hudStrategy = document.getElementById('hud-strategy');
    const hudLatency = document.getElementById('hud-latency');
    const hudBreaker = document.getElementById('hud-breaker');
    const hudCandidates = document.getElementById('hud-candidates');

    // Inspector
    const inspectorDrawer = document.getElementById('inspector-drawer');
    const btnToggleInspector = document.getElementById('btn-toggle-inspector');
    const btnCloseDrawer = document.getElementById('btn-close-drawer');
    const jsonViewer = document.getElementById('inspector-json-viewer');

    // Modals
    const articleModal = document.getElementById('article-modal-backdrop');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const modalCategory = document.getElementById('modal-category');
    const modalReason = document.getElementById('modal-reason');
    const modalTitle = document.getElementById('modal-title');
    const modalNewsId = document.getElementById('modal-news-id');
    const modalScore = document.getElementById('modal-score');
    const modalAbstract = document.getElementById('modal-abstract');
    const btnModalAddHistory = document.getElementById('btn-modal-add-history');
    const btnModalOpenUrl = document.getElementById('btn-modal-open-url');
    let currentModalArticle = null;

    // Custom User Modal
    const customUserModal = document.getElementById('custom-user-modal-backdrop');
    const btnCustomUser = document.getElementById('btn-custom-user');
    const btnCloseCustomModal = document.getElementById('btn-close-custom-modal');
    const btnCancelCustomUser = document.getElementById('btn-cancel-custom-user');
    const btnApplyCustomUser = document.getElementById('btn-apply-custom-user');
    const customInputUserId = document.getElementById('custom-input-user-id');
    const customInputHistory = document.getElementById('custom-input-history');

    // Toast
    const toast = document.getElementById('app-toast');

    function showToast(message) {
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2600);
    }

    // Map Category to CSS Class
    function getCategoryClass(category) {
        const cat = (category || '').toLowerCase();
        const supported = ['sports', 'news', 'finance', 'lifestyle', 'health', 'foodanddrink', 'travel', 'tv'];
        return supported.includes(cat) ? `cat-${cat}` : 'cat-general';
    }

    // Map Reason Code to Human Friendly Vietnamese Label & Icon
    function formatReason(reasonCode) {
        switch (reasonCode) {
            case 'MATCHES_YOUR_TASTE':
                return {
                    label: 'Phù hợp sở thích (CF Index)',
                    icon: '🎯',
                    cls: 'reason-taste',
                    desc: 'Tìm kiếm qua vector FAISS người dùng'
                };
            case 'SIMILAR_TO_RECENT_HISTORY':
                return {
                    label: 'Tương tự bài vừa đọc (Content)',
                    icon: '⚡',
                    cls: 'reason-history',
                    desc: 'Khớp nội dung văn bản qua TF-IDF'
                };
            case 'CONTINUES_RECENT_INTEREST':
                return {
                    label: 'Tiếp nối chuỗi quan tâm (SASRec)',
                    icon: '🔄',
                    cls: 'reason-seq',
                    desc: 'Mô hình Transformer tuần tự tự chú ý'
                };
            case 'POPULAR_NOW':
            default:
                return {
                    label: 'Tin nóng thịnh hành (Trending)',
                    icon: '🔥',
                    cls: 'reason-popular',
                    desc: 'Cứu cánh an toàn khi thiếu dữ liệu'
                };
        }
    }

    // 1. Fetch System Health & Readiness
    async function checkHealth() {
        try {
            const res = await fetch('/ready');
            if (res.ok) {
                const data = await res.json();
                healthPill.className = 'status-pill status-healthy';
                healthText.textContent = `Hệ Thống Sẵn Sàng (${data.model_version || 'dev'})`;
            } else {
                healthPill.className = 'status-pill status-degraded';
                healthText.textContent = 'Đang Khởi Tạo Bundle...';
            }
        } catch {
            healthPill.className = 'status-pill status-degraded';
            healthText.textContent = 'API Đang Offline';
        }
    }

    // 2. Fetch Personas
    async function loadPersonas() {
        try {
            const res = await fetch('/api/personas');
            if (res.ok) {
                currentPersonas = await res.json();
                renderPersonasGrid();
                if (currentPersonas.length > 0) {
                    selectPersona(currentPersonas[0]);
                }
            }
        } catch (e) {
            console.error('Failed to load personas', e);
        }
    }

    function renderPersonasGrid() {
        personasGrid.innerHTML = '';
        currentPersonas.forEach((p, idx) => {
            const card = document.createElement('div');
            card.className = `persona-card ${selectedPersona && selectedPersona.id === p.id ? 'active' : ''}`;
            card.id = `persona-card-${p.id}`;
            card.innerHTML = `
                <div class="persona-top-row">
                    <div class="persona-avatar">${p.avatar}</div>
                    <div>
                        <div class="persona-name">${p.name}</div>
                        <div class="persona-role">${p.role}</div>
                    </div>
                </div>
                <p class="persona-desc">${p.description}</p>
                <div class="persona-meta">
                    <span>ID: ${p.id}</span>
                    <span>${p.history ? p.history.length : 0} bài đã đọc</span>
                </div>
            `;
            card.addEventListener('click', () => selectPersona(p));
            personasGrid.appendChild(card);
        });
    }

    function selectPersona(persona) {
        selectedPersona = persona;
        currentHistoryIds = [...(persona.history_news_ids || [])];
        
        // Highlight active card
        document.querySelectorAll('.persona-card').forEach(c => c.classList.remove('active'));
        const activeCard = document.getElementById(`persona-card-${persona.id}`);
        if (activeCard) activeCard.classList.add('active');

        // Update profile section
        userAvatar.textContent = persona.avatar;
        userName.textContent = persona.name;
        userIdBadge.textContent = `ID: ${persona.id}`;
        userRole.textContent = persona.role;
        userDesc.textContent = persona.description;

        renderHistoryChips(persona.history || []);
        fetchRecommendations();
    }

    function renderHistoryChips(historyList) {
        historyChips.innerHTML = '';
        if (!currentHistoryIds || currentHistoryIds.length === 0) {
            historyChips.innerHTML = '<span class="text-dim" style="font-size: 12px;">(Chưa có bài đọc nào — Độc giả mới tinh)</span>';
            return;
        }

        currentHistoryIds.forEach(nid => {
            const matched = (historyList || []).find(h => h.news_id === nid);
            const title = matched ? matched.title : `Bài viết ${nid}`;
            const cat = matched ? matched.category : 'news';
            
            const chip = document.createElement('div');
            chip.className = 'history-chip';
            chip.innerHTML = `
                <span class="chip-cat ${getCategoryClass(cat)}">${cat}</span>
                <span>${title}</span>
            `;
            historyChips.appendChild(chip);
        });
    }

    // 3. Clear History Button
    btnClearHistory.addEventListener('click', () => {
        currentHistoryIds = [];
        renderHistoryChips([]);
        showToast('Đã xóa lịch sử đọc! Chế độ Cold User được kích hoạt.');
        fetchRecommendations();
    });

    // 4. Fetch Recommendations from Backend
    async function fetchRecommendations() {
        const topK = parseInt(topkSelect.value, 10) || 10;
        const filterSeen = filterSeenToggle.checked;
        const userId = selectedPersona ? selectedPersona.id : 'U134050';

        btnRecommend.disabled = true;
        btnRecommend.querySelector('.btn-text').textContent = 'Đang Xếp Hạng...';
        newsGrid.innerHTML = `
            <div class="skeleton" style="border-radius: 16px;"></div>
            <div class="skeleton" style="border-radius: 16px;"></div>
            <div class="skeleton" style="border-radius: 16px;"></div>
        `;

        const startTime = performance.now();

        try {
            const payload = {
                user_id: userId === 'NEW_USER_COLD' ? null : userId,
                history_news_ids: currentHistoryIds,
                top_k: topK,
                filter_seen: filterSeen,
                context: {}
            };

            const res = await fetch('/api/recommend-rich', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const elapsed = Math.round(performance.now() - startTime);

            if (!res.ok) {
                throw new Error(`HTTP Error ${res.status}`);
            }

            const data = await res.json();
            currentRecommendations = data.recommendations || [];

            // Update HUD
            hudStrategy.textContent = data.strategy ? data.strategy.toUpperCase() : 'HYBRID';
            hudLatency.textContent = `${data.latency_ms ? data.latency_ms.toFixed(1) : elapsed} ms`;
            
            if (data.degraded) {
                hudBreaker.textContent = `Hạ cấp (${data.fallback_reason || 'COLD_USER'})`;
                hudBreaker.className = 'hud-value highlight-cyan';
            } else {
                hudBreaker.textContent = 'Hoạt động tối ưu (Normal)';
                hudBreaker.className = 'hud-value highlight-green';
            }

            // Update Inspector
            jsonViewer.textContent = JSON.stringify(data, null, 2);

            renderNewsGrid();
        } catch (err) {
            console.error('Recommendation failed', err);
            newsGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 40px; background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 16px;">
                    <div style="font-size: 32px; margin-bottom: 10px;">⚠️</div>
                    <h3>Không thể tải gợi ý lúc này</h3>
                    <p style="color: #94a3b8; font-size: 13px;">Chi tiết lỗi: ${err.message}</p>
                </div>
            `;
        } finally {
            btnRecommend.disabled = false;
            btnRecommend.querySelector('.btn-text').textContent = 'Gợi Ý Tin Tức Mới';
        }
    }

    // 5. Render News Cards Grid
    function renderNewsGrid() {
        newsGrid.innerHTML = '';

        const filtered = activeCategoryFilter === 'all'
            ? currentRecommendations
            : currentRecommendations.filter(item => {
                const cat = (item.article?.category || '').toLowerCase();
                return cat === activeCategoryFilter;
            });

        if (filtered.length === 0) {
            newsGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 48px; color: #94a3b8;">
                    <p style="font-size: 16px;">Không tìm thấy bài viết nào trong chuyên mục "${activeCategoryFilter}".</p>
                    <button class="btn btn-outline" style="margin-top: 14px;" onclick="document.querySelector('.filter-pill[data-cat=all]').click()">Xem tất cả thể loại</button>
                </div>
            `;
            return;
        }

        filtered.forEach(item => {
            const article = item.article || {
                title: `Bài viết ${item.news_id}`,
                category: 'news',
                abstract: 'Tóm tắt tin tức...',
                url: '#'
            };
            const reason = formatReason(item.reason_code);
            const catCls = getCategoryClass(article.category);
            const scoreDisplay = item.score ? item.score.toFixed(4) : 'N/A';

            const card = document.createElement('div');
            card.className = 'news-card';
            card.innerHTML = `
                <div>
                    <div class="news-card-top">
                        <span class="rank-badge ${item.rank === 1 ? 'top-1' : ''}">Top #${item.rank}</span>
                        <span class="category-tag ${catCls}">${article.category}</span>
                    </div>
                    <h3 class="news-title" title="${article.title}">${article.title}</h3>
                    <p class="news-abstract">${article.abstract}</p>
                </div>

                <div>
                    <div class="reason-box">
                        <div class="reason-tag ${reason.cls}">
                            <span>${reason.icon}</span>
                            <span>${reason.label}</span>
                        </div>
                        <div class="reason-subtext">${reason.desc} • Điểm: ${scoreDisplay}</div>
                    </div>

                    <div class="news-card-actions">
                        <button class="btn btn-read" data-id="${item.news_id}">Đọc Bài</button>
                        <button class="btn btn-simulate-click" data-id="${item.news_id}" title="Thêm bài này vào lịch sử đọc để xem AI phản hồi">+ Đọc tiếp</button>
                    </div>
                </div>
            `;

            // Read Button
            card.querySelector('.btn-read').addEventListener('click', () => openArticleModal(item));

            // Simulate Click
            card.querySelector('.btn-simulate-click').addEventListener('click', () => {
                if (!currentHistoryIds.includes(item.news_id)) {
                    currentHistoryIds.push(item.news_id);
                    renderHistoryChips(selectedPersona?.history || []);
                    showToast(`Đã thêm "${article.title.substring(0, 30)}..." vào lịch sử click!`);
                    fetchRecommendations();
                } else {
                    showToast('Bài này đã có trong lịch sử đọc rồi.');
                }
            });

            newsGrid.appendChild(card);
        });
    }

    // 6. Category Filter Pills
    filterPills.forEach(pill => {
        pill.addEventListener('click', () => {
            filterPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            activeCategoryFilter = pill.getAttribute('data-cat');
            renderNewsGrid();
        });
    });

    // 7. Modal Handlers
    function openArticleModal(item) {
        currentModalArticle = item;
        const art = item.article || {};
        const reason = formatReason(item.reason_code);

        modalCategory.textContent = (art.category || 'NEWS').toUpperCase();
        modalCategory.className = `badge ${getCategoryClass(art.category)}`;
        modalReason.textContent = `${reason.icon} ${reason.label}`;
        modalReason.className = `badge ${reason.cls}`;
        
        modalTitle.textContent = art.title || 'Tiêu đề bài viết';
        modalNewsId.textContent = `Mã bài: ${item.news_id}`;
        modalScore.textContent = `Độ liên quan xếp hạng: ${item.score ? item.score.toFixed(4) : 'N/A'}`;
        modalAbstract.textContent = art.abstract || 'Không có tóm tắt chi tiết cho bài viết này.';

        btnModalOpenUrl.href = art.url && art.url !== '#' ? art.url : `https://assets.msn.com/labs/mind/${item.news_id}.html`;
        articleModal.classList.add('open');
    }

    btnCloseModal.addEventListener('click', () => articleModal.classList.remove('open'));
    articleModal.addEventListener('click', (e) => {
        if (e.target === articleModal) articleModal.classList.remove('open');
    });

    btnModalAddHistory.addEventListener('click', () => {
        if (currentModalArticle && !currentHistoryIds.includes(currentModalArticle.news_id)) {
            currentHistoryIds.push(currentModalArticle.news_id);
            renderHistoryChips(selectedPersona?.history || []);
            articleModal.classList.remove('open');
            showToast('Đã thêm bài vào lịch sử đọc!');
            fetchRecommendations();
        }
    });

    // 8. Custom User Dialog
    btnCustomUser.addEventListener('click', () => {
        customUserModal.classList.add('open');
    });
    btnCloseCustomModal.addEventListener('click', () => customUserModal.classList.remove('open'));
    btnCancelCustomUser.addEventListener('click', () => customUserModal.classList.remove('open'));

    btnApplyCustomUser.addEventListener('click', () => {
        const uid = customInputUserId.value.trim() || 'U_CUSTOM';
        const rawHistory = customInputHistory.value.split(',').map(s => s.trim()).filter(Boolean);

        const customPersona = {
            id: uid,
            name: `Người Dùng ${uid}`,
            role: 'Custom Reader Profile',
            avatar: '👤',
            history_news_ids: rawHistory,
            history: rawHistory.map(nid => ({ news_id: nid, title: `Bài ${nid}`, category: 'custom' })),
            description: 'Hồ sơ người dùng tùy biến nhập tay từ giao diện.'
        };

        customUserModal.classList.remove('open');
        selectPersona(customPersona);
        showToast(`Đã áp dụng cấu hình người dùng ${uid}!`);
    });

    // 9. Inspector Drawer Handlers
    btnToggleInspector.addEventListener('click', () => {
        inspectorDrawer.classList.toggle('open');
    });
    btnCloseDrawer.addEventListener('click', () => {
        inspectorDrawer.classList.remove('open');
    });

    // 10. Run Recommendation Button Click
    btnRecommend.addEventListener('click', fetchRecommendations);

    // Initial sequence
    checkHealth();
    loadPersonas();
});

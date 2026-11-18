/* Reader controls and news feed. Recommendation data comes from the serving API. */
'use strict';

document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);
    // label, emoji, hue (for the colored category tag)
    const categories = {
        sports: ['Thể thao', '⚽', 140], news: ['Thời sự', '📰', 215], finance: ['Tài chính', '💹', 170],
        health: ['Sức khỏe', '🩺', 350], lifestyle: ['Đời sống', '🌿', 95], foodanddrink: ['Ẩm thực', '🍜', 25],
        travel: ['Du lịch', '✈️', 195], tv: ['Truyền hình', '📺', 265], entertainment: ['Giải trí', '🎭', 300],
        movies: ['Điện ảnh', '🎬', 280], music: ['Âm nhạc', '🎵', 320], autos: ['Xe', '🚗', 230],
        weather: ['Thời tiết', '🌦️', 200], video: ['Video', '▶️', 5], kids: ['Gia đình', '🧸', 45],
        northamerica: ['Bắc Mỹ', '🌎', 180], middleeast: ['Trung Đông', '🌍', 35], general: ['Tổng hợp', '🗞️', 40],
    };
    const reasons = {
        MATCHES_YOUR_TASTE: ['✨', 'Hợp với sở thích của bạn'],
        SIMILAR_TO_RECENT_HISTORY: ['🔁', 'Giống những bài bạn vừa đọc'],
        CONTINUES_RECENT_INTEREST: ['📌', 'Tiếp nối chủ đề bạn đang theo dõi'],
        TRENDING_IN_CATEGORY: ['📈', 'Đang nổi trong chuyên mục'],
        POPULAR_NOW: ['🔥', 'Nhiều người đang đọc'],
    };
    const avatarHues = [140, 25, 215, 350, 265, 45, 195, 300, 95, 5, 170, 230];
    const state = {
        personas: [], selected: null, history: [], articles: new Map(), recommendations: [],
        category: 'all', response: null, httpMs: null, controller: null, requestId: 0,
        updatedAt: null, modalArticle: null, topK: 10,
    };
    let toastTimer;
    let optionTimer;
    const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[char]));
    const categoryInfo = (key) => {
        const [label, emoji, hue] = categories[key] || [key || 'Tổng hợp', '🗞️', 40];
        return { label, emoji, hue };
    };
    const categoryName = (key) => categoryInfo(key).label;
    const categoryTag = (key) => {
        const { label, emoji, hue } = categoryInfo(key);
        return `<span class="category-tag" style="--h:${hue}"><span class="cat-emoji" aria-hidden="true">${emoji}</span>${escape(label)}</span>`;
    };
    const reasonInfo = (code) => reasons[code] || ['💡', 'Một gợi ý dành cho bạn'];
    const reasonPill = (code) => {
        const [icon, label] = reasonInfo(code);
        return `<span class="reason-icon" aria-hidden="true">${icon}</span>${escape(label)}`;
    };
    const isGuest = (persona) => !persona?.id || persona.id === 'NEW_USER_COLD';
    const readerName = (persona) => isGuest(persona) && !persona?.custom ? 'Người đọc mới' : persona.name;
    const firstName = (persona) => {
        if (persona.custom || isGuest(persona)) return 'bạn';
        return readerName(persona).trim().split(/\s+/).slice(-2).join(' ');
    };
    const avatarOf = (persona) => persona.avatar || (persona.custom ? '🙂' : '👤');
    const narrowScreen = window.matchMedia('(max-width: 820px)');

    const today = new Date().toLocaleDateString('vi-VN', { weekday: 'long', day: 'numeric', month: 'long' });
    $('hero-date').textContent = today;

    function toast(message) {
        clearTimeout(toastTimer);
        $('toast').textContent = message;
        $('toast').dataset.visible = 'true';
        toastTimer = setTimeout(() => { $('toast').dataset.visible = 'false'; }, 3200);
    }

    function setConnection(status, label) {
        $('connection-status').dataset.state = status;
        $('connection-label').textContent = label;
        $('connection-status').title = label;
    }

    async function checkReady() {
        try {
            const response = await fetch('/ready');
            setConnection(response.ok ? 'ready' : 'offline', response.ok ? 'Hệ thống sẵn sàng' : 'Đang khởi động');
        } catch {
            setConnection('offline', 'Mất kết nối');
        }
    }

    function renderReaders() {
        $('reader-list').innerHTML = state.personas.map((persona, index) => {
            const active = state.selected === persona;
            const interest = isGuest(persona) && !persona.custom
                ? 'Chưa đọc bài nào'
                : persona.custom ? `${persona.history_news_ids.length} bài tự nhập` : persona.category_pref || 'Hồ sơ tùy chọn';
            const hue = avatarHues[index % avatarHues.length];
            return `<button class="reader-option" type="button" data-reader="${index}" aria-pressed="${active}">
                <span class="reader-avatar" style="--h:${hue}" aria-hidden="true">${escape(avatarOf(persona))}</span>
                <span class="reader-option-copy"><span class="reader-name">${escape(readerName(persona))}</span><span class="reader-interest">${escape(interest)}</span></span>
                ${active ? '<span class="reader-check" aria-hidden="true">✓</span>' : ''}
            </button>`;
        }).join('');
    }

    function rememberArticles(articles) {
        for (const article of articles || []) {
            if (article.news_id) state.articles.set(article.news_id, article);
        }
    }

    function renderHero() {
        const persona = state.selected;
        if (!persona) return;
        $('feed-heading').textContent = `Chào ${firstName(persona)} 👋`;
        $('hero-avatar').textContent = avatarOf(persona);
        $('feed-description').textContent = !state.history.length
            ? 'Bạn chưa đọc bài nào, nên mình bắt đầu với những bài đang được nhiều người quan tâm nhé.'
            : `Dựa trên ${state.history.length} bài bạn đã đọc, đây là những bài mình nghĩ bạn sẽ thích.`;
        const counts = new Map();
        for (const id of state.history) {
            const category = state.articles.get(id)?.category;
            if (category && category !== 'general') counts.set(category, (counts.get(category) || 0) + 1);
        }
        const top = [...counts].sort((a, b) => b[1] - a[1]).slice(0, 4);
        $('interest-row').hidden = !top.length;
        $('interest-chips').innerHTML = top.map(([key]) => {
            const { label, emoji } = categoryInfo(key);
            return `<span class="interest-chip"><span aria-hidden="true">${emoji}</span>${escape(label)}</span>`;
        }).join('');
    }

    function selectReader(persona) {
        state.selected = persona;
        state.history = [...(persona.history_news_ids || [])].slice(-500);
        rememberArticles(persona.history);
        state.category = 'all';
        state.recommendations = [];
        state.response = null;
        renderReaders();
        renderHistory();
        renderHero();
        renderDetails();
        fetchRecommendations();
    }

    function renderHistory() {
        $('history-count').textContent = state.history.length;
        $('history-context').textContent = state.selected
            ? (state.history.length ? 'Những bài này giúp hệ thống hiểu gu của bạn.' : 'Lịch sử đang trống.')
            : 'Chọn người đọc để xem lịch sử.';
        $('clear-history-button').hidden = state.history.length === 0;
        if (!state.history.length) {
            $('history-list').innerHTML = '<li class="history-empty">Chưa có bài nào.<br>Bấm “Đánh dấu đã đọc” ở một bài bất kỳ để bắt đầu.</li>';
            return;
        }
        // Show the eight most recent articles; send the entire history to the API.
        const shown = state.history.slice(-8).reverse();
        const rows = shown.map((id) => {
            const article = state.articles.get(id);
            const title = article?.title || `Bài viết ${id}`;
            const { label, emoji, hue } = categoryInfo(article?.category);
            return `<li class="history-item"><span class="history-item-icon" style="--h:${hue}" aria-hidden="true">${emoji}</span>
                <div class="history-item-copy"><p class="history-item-title" title="${escape(title)}">${escape(title)}</p><span class="history-item-cat">${escape(label)}</span></div>
                <button class="remove-history" type="button" data-remove-history="${escape(id)}" aria-label="Bỏ ${escape(title)} khỏi lịch sử" title="Bỏ khỏi lịch sử">×</button></li>`;
        });
        const more = state.history.length - shown.length;
        if (more > 0) rows.push(`<li class="history-more">và ${more} bài khác</li>`);
        $('history-list').innerHTML = rows.join('');
    }

    function showNotice(kind, icon, text, action = false) {
        $('feed-notice').dataset.kind = kind;
        $('feed-notice-icon').textContent = icon;
        $('feed-notice-text').textContent = text;
        $('notice-refresh').hidden = !action;
        $('feed-notice').hidden = false;
    }

    function markStale() {
        showNotice('stale', '✏️', 'Lịch sử đọc vừa thay đổi. Cập nhật để xem gợi ý mới theo sở thích hiện tại.', true);
    }

    function invalidateRequest() {
        state.requestId += 1;
        state.controller?.abort();
        $('recommend-button').disabled = !state.selected;
        $('recommend-button').removeAttribute('aria-busy');
        $('recommend-button-label').textContent = 'Làm mới gợi ý';
        $('article-list').setAttribute('aria-busy', 'false');
    }

    function updateHistory(nextHistory) {
        invalidateRequest();
        state.history = nextHistory;
        renderHistory();
        renderHero();
        renderNews();
        markStale();
    }

    function addToHistory(item) {
        if (state.history.includes(item.news_id)) {
            toast('Bài này đã có trong lịch sử đọc rồi.');
            return;
        }
        rememberArticles([{ ...item.article, news_id: item.news_id }]);
        updateHistory([...state.history, item.news_id].slice(-500));
        toast('✓ Đã đánh dấu là đã đọc');
    }

    function loadingState() {
        $('article-list').innerHTML = Array.from({ length: 5 }, () => '<div class="loading-article" aria-hidden="true"><div class="loading-line short"></div><div class="loading-line title"></div><div class="loading-line"></div><div class="loading-line" style="width:70%"></div></div>').join('');
        $('article-list').setAttribute('aria-busy', 'true');
    }

    function emptyState(icon, title, description, retry = false) {
        $('article-list').innerHTML = `<div class="empty-state"><span class="empty-state-mark" aria-hidden="true">${icon}</span><h2>${escape(title)}</h2><p>${escape(description)}</p>${retry ? '<button class="primary-button" type="button" id="retry-button">Thử lại</button>' : ''}</div>`;
        $('retry-button')?.addEventListener('click', fetchRecommendations);
    }

    function renderNotice() {
        const result = state.response;
        $('feed-notice').hidden = true;
        if (!result) return;
        if (result.fallback_reason === 'COLD_USER') {
            showNotice('info', '👋', 'Hệ thống chưa biết nhiều về hồ sơ này, nên đang gợi ý dựa trên các bài phổ biến và lịch sử bạn cung cấp.');
        } else if (result.degraded) {
            showNotice('warning', '⚠️', 'Một phần hệ thống đang phản hồi chậm, bảng tin dùng kết quả dự phòng. Bạn có thể thử làm mới lại.', true);
        }
    }

    async function fetchRecommendations() {
        if (!state.selected) return;
        clearTimeout(optionTimer);
        state.controller?.abort();
        const controller = new AbortController();
        state.controller = controller;
        const requestId = ++state.requestId;
        $('recommend-button').disabled = true;
        $('recommend-button').setAttribute('aria-busy', 'true');
        $('recommend-button-label').textContent = 'Đang tìm bài…';
        $('feed-notice').hidden = true;
        $('category-nav').hidden = true;
        $('feed-caption').textContent = 'Đang chọn những bài hợp với bạn…';
        loadingState();
        const started = performance.now();
        try {
            const response = await fetch('/api/recommend-rich', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
                body: JSON.stringify({ user_id: isGuest(state.selected) ? null : state.selected.id,
                    history_news_ids: [...state.history], top_k: state.topK,
                    filter_seen: $('filter-seen').checked, context: {} }),
            });
            if (!response.ok) throw new Error(response.status === 503 ? 'not-ready' : `http-${response.status}`);
            const result = await response.json();
            if (requestId !== state.requestId) return;
            if (!Array.isArray(result.recommendations)) throw new Error('invalid-response');
            state.response = result;
            state.httpMs = Math.round(performance.now() - started);
            state.updatedAt = new Date();
            state.recommendations = result.recommendations;
            rememberArticles(state.recommendations.map((item) => ({ ...item.article, news_id: item.news_id })));
            state.category = 'all';
            setConnection('ready', 'Hệ thống sẵn sàng');
            renderCategories();
            renderNews();
            renderNotice();
            renderDetails();
        } catch (error) {
            if (error.name === 'AbortError' || requestId !== state.requestId) return;
            state.response = null;
            state.recommendations = [];
            $('feed-caption').textContent = 'Bảng tin chưa được cập nhật';
            const description = error.message === 'not-ready'
                ? 'Hệ thống vẫn đang nạp dữ liệu gợi ý. Bạn đợi một chút rồi thử lại nhé.'
                : 'Chưa kết nối được tới máy chủ gợi ý. Hãy kiểm tra server rồi thử lại.';
            emptyState('🔌', 'Chưa lấy được bảng tin', description, true);
            renderDetails();
            checkReady();
        } finally {
            if (requestId === state.requestId) {
                $('recommend-button').disabled = false;
                $('recommend-button').removeAttribute('aria-busy');
                $('recommend-button-label').textContent = 'Làm mới gợi ý';
                $('article-list').setAttribute('aria-busy', 'false');
            }
        }
    }

    function renderCategories() {
        const counts = new Map();
        for (const item of state.recommendations) {
            const category = item.article?.category || 'general';
            counts.set(category, (counts.get(category) || 0) + 1);
        }
        const buttons = [['all', 'Tất cả', '', state.recommendations.length],
            ...[...counts].sort((a, b) => b[1] - a[1]).map(([key, count]) => [key, categoryName(key), categoryInfo(key).emoji, count])];
        $('category-nav').innerHTML = buttons.map(([key, label, emoji, count]) => `<button class="category-button" type="button" data-category="${escape(key)}" aria-pressed="${state.category === key}">${emoji ? `<span aria-hidden="true">${emoji}</span>` : ''}${escape(label)}<span class="cat-count">${count}</span></button>`).join('');
        $('category-nav').hidden = counts.size < 2;
    }

    function renderNews() {
        const items = state.recommendations.filter((item) => state.category === 'all' || (item.article?.category || 'general') === state.category);
        const time = state.updatedAt?.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
        $('feed-caption').textContent = state.response
            ? `${items.length} bài${state.category !== 'all' ? ` · ${categoryName(state.category)}` : ''}${time ? ` · cập nhật lúc ${time}` : ''}`
            : 'Chưa có lượt gợi ý';
        if (!items.length) {
            if (state.response) emptyState('🔍', 'Chưa có bài phù hợp', 'Thử chọn chuyên mục khác, đánh dấu thêm vài bài đã đọc hoặc làm mới gợi ý.');
            else emptyState('🗞️', 'Bảng tin đang chờ', 'Bấm “Làm mới gợi ý” để xem những bài dành cho bạn.');
            return;
        }
        $('article-list').innerHTML = items.map((item, index) => {
            const article = item.article || {};
            const saved = state.history.includes(item.news_id);
            const featured = index === 0 && state.category === 'all';
            const sub = article.subcategory && article.subcategory !== article.category ? `<span class="article-sub">${escape(article.subcategory)}</span>` : '';
            return `<article class="news-article" style="animation-delay:${Math.min(index, 8) * 35}ms">
                <div class="article-meta">${categoryTag(article.category)}${sub}${featured ? '<span class="featured-label">⭐ Hợp nhất với bạn</span>' : ''}<span class="article-rank" title="Thứ hạng gợi ý">#${item.rank}</span></div>
                <h2 class="article-title"><button class="article-title-button" type="button" data-open-article="${escape(item.news_id)}">${escape(article.title || `Bài viết ${item.news_id}`)}</button></h2>
                <p class="article-abstract">${escape(article.abstract || 'Bài viết này chưa có nội dung tóm tắt.')}</p>
                <div class="article-bottom"><p class="reason-pill">${reasonPill(item.reason_code)}</p>
                    <button class="save-history-button" type="button" data-save-article="${escape(item.news_id)}" data-saved="${saved}">${saved ? '✓ Đã đọc' : '＋ Đánh dấu đã đọc'}</button>
                </div></article>`;
        }).join('');
    }

    function renderDetails() {
        const result = state.response;
        const rows = result ? [
            ['Người đọc', isGuest(state.selected) ? 'Người đọc mới' : state.selected.id],
            ['Phiên bản model', result.model_version || '—'],
            ['Cách gợi ý', result.strategy === 'hybrid' ? 'Kết hợp nhiều nguồn' : result.strategy === 'popular' ? 'Theo mức độ phổ biến' : result.strategy || '—'],
            ['Thời gian trong pipeline', Number.isFinite(result.latency_ms) ? `${result.latency_ms.toFixed(1)} ms` : '—'],
            ['Thời gian HTTP', `${state.httpMs} ms`],
            ['Số bài trả về', result.recommendations.length],
            ['Trạng thái', result.degraded ? `Dùng dự phòng (${result.fallback_reason || 'không rõ'})` : 'Hoàn tất'],
        ] : [['Trạng thái', 'Chưa có kết quả của hồ sơ hiện tại']];
        $('request-details').innerHTML = rows.map(([label, value]) => `<dt>${escape(label)}</dt><dd>${escape(value)}</dd>`).join('');
        $('response-json').textContent = result ? JSON.stringify(result, null, 2) : 'Chưa có phản hồi.';
    }

    function syncMarkReadButton() {
        const saved = state.modalArticle && state.history.includes(state.modalArticle.news_id);
        $('mark-read-button').textContent = saved ? '✓ Đã có trong lịch sử' : 'Đánh dấu đã đọc';
        $('mark-read-button').disabled = Boolean(saved);
    }

    function openArticle(item) {
        state.modalArticle = item;
        const article = item.article || {};
        $('article-dialog-title').textContent = article.title || `Bài viết ${item.news_id}`;
        $('article-dialog-category').innerHTML = categoryTag(article.category)
            + (article.subcategory ? `<span class="article-sub">${escape(article.subcategory)}</span>` : '');
        $('article-dialog-reason').innerHTML = reasonPill(item.reason_code);
        $('article-dialog-abstract').textContent = article.abstract || 'Bài viết này chưa có nội dung tóm tắt.';
        syncMarkReadButton();
        const link = $('original-article-link');
        let url;
        try {
            const parsed = new URL(article.url);
            if (['https:', 'http:'].includes(parsed.protocol)) url = parsed.href;
        } catch { /* Metadata may not contain an original URL. */ }
        link.hidden = !url;
        if (url) link.href = url;
        else link.removeAttribute('href');
        $('article-dialog').showModal();
    }

    function scheduleOptionRefresh() {
        if (!state.selected) return;
        clearTimeout(optionTimer);
        optionTimer = setTimeout(fetchRecommendations, 250);
    }

    $('reader-list').addEventListener('click', (event) => {
        const button = event.target.closest('[data-reader]');
        if (!button) return;
        const persona = state.personas[Number(button.dataset.reader)];
        if (persona === state.selected) return;
        selectReader(persona);
        if (narrowScreen.matches) $('news-feed').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    $('history-list').addEventListener('click', (event) => {
        const button = event.target.closest('[data-remove-history]');
        if (button) updateHistory(state.history.filter((id) => id !== button.dataset.removeHistory));
    });
    $('clear-history-button').addEventListener('click', () => {
        updateHistory([]);
        toast('Đã xóa lịch sử của lượt trải nghiệm này.');
    });
    $('recommend-button').addEventListener('click', fetchRecommendations);
    $('notice-refresh').addEventListener('click', fetchRecommendations);
    $('article-count').addEventListener('click', (event) => {
        const button = event.target.closest('[data-count]');
        if (!button) return;
        state.topK = Number(button.dataset.count);
        for (const option of $('article-count').querySelectorAll('[data-count]')) {
            option.setAttribute('aria-checked', String(option === button));
        }
        scheduleOptionRefresh();
    });
    $('article-count').addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        const options = [...$('article-count').querySelectorAll('[data-count]')];
        const current = options.findIndex((option) => option.getAttribute('aria-checked') === 'true');
        const next = options[(current + (event.key === 'ArrowRight' ? 1 : options.length - 1)) % options.length];
        next.focus();
        next.click();
        event.preventDefault();
    });
    $('filter-seen').addEventListener('change', scheduleOptionRefresh);
    $('category-nav').addEventListener('click', (event) => {
        const button = event.target.closest('[data-category]');
        if (!button) return;
        state.category = button.dataset.category;
        renderCategories();
        renderNews();
    });
    $('article-list').addEventListener('click', (event) => {
        const button = event.target.closest('[data-open-article], [data-save-article]');
        if (!button) return;
        const id = button.dataset.openArticle || button.dataset.saveArticle;
        const item = state.recommendations.find((entry) => entry.news_id === id);
        if (!item) return;
        if (button.hasAttribute('data-open-article')) openArticle(item);
        else addToHistory(item);
    });
    $('mark-read-button').addEventListener('click', () => {
        if (state.modalArticle) addToHistory(state.modalArticle);
        $('article-dialog').close();
    });
    $('custom-reader-button').addEventListener('click', () => {
        $('custom-form-error').hidden = true;
        $('custom-dialog').showModal();
        $('custom-user-id').focus();
    });
    $('custom-reader-form').addEventListener('submit', (event) => {
        event.preventDefault();
        const history = $('custom-history').value.trim().split(/[\s,]+/).filter(Boolean);
        if (history.length > 500) {
            $('custom-form-error').textContent = 'Bạn nhập tối đa 500 mã bài cho một lượt gợi ý nhé.';
            $('custom-form-error').hidden = false;
            return;
        }
        const id = $('custom-user-id').value.trim();
        const persona = { id: id || null, name: 'Hồ sơ của bạn', avatar: '🙂', history_news_ids: history, history: [], custom: true };
        state.personas = state.personas.filter((entry) => !entry.custom);
        state.personas.push(persona);
        $('custom-dialog').close();
        selectReader(persona);
    });
    $('details-button').addEventListener('click', () => $('details-dialog').showModal());
    $('help-button').addEventListener('click', () => $('help-dialog').showModal());
    for (const dialog of document.querySelectorAll('dialog')) {
        dialog.querySelectorAll('[data-close-dialog]').forEach((button) => button.addEventListener('click', () => dialog.close()));
        dialog.addEventListener('click', (event) => {
            const bounds = dialog.getBoundingClientRect();
            if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
        });
    }

    function showHelpOnFirstVisit() {
        try {
            if (localStorage.getItem('recsys-help-seen')) return;
            localStorage.setItem('recsys-help-seen', '1');
        } catch { return; }
        $('help-dialog').showModal();
    }

    async function initialize() {
        checkReady();
        try {
            const response = await fetch('/api/personas');
            if (!response.ok) throw new Error('personas-unavailable');
            const personas = await response.json();
            if (!Array.isArray(personas) || !personas.length) throw new Error('no-personas');
            state.personas = personas;
            for (const persona of personas) rememberArticles(persona.history);
            selectReader(personas[0]);
        } catch {
            $('reader-list').innerHTML = '<p class="history-empty">Chưa tải được hồ sơ mẫu. Bạn vẫn có thể “Tự nhập hồ sơ của bạn”.</p>';
            $('feed-caption').textContent = 'Chưa tải được hồ sơ';
            $('article-list').setAttribute('aria-busy', 'false');
            emptyState('🧭', 'Chọn cách bắt đầu', 'Tự nhập mã người đọc, hoặc để trống để thử như một người đọc mới.');
        }
        showHelpOnFirstVisit();
    }
    initialize();
});

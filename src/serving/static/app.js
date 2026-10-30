/* Reader controls and news feed. Recommendation data comes from the serving API. */
'use strict';

document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);
    const categories = {
        sports: 'Thể thao', news: 'Thời sự', finance: 'Tài chính', health: 'Sức khỏe',
        lifestyle: 'Đời sống', foodanddrink: 'Ẩm thực', travel: 'Du lịch', tv: 'Truyền hình',
        entertainment: 'Giải trí', movies: 'Điện ảnh', music: 'Âm nhạc', autos: 'Xe',
        weather: 'Thời tiết', video: 'Video', kids: 'Gia đình', general: 'Tổng hợp',
    };
    const reasons = {
        MATCHES_YOUR_TASTE: 'Phù hợp với sở thích của bạn',
        SIMILAR_TO_RECENT_HISTORY: 'Gần với những bài bạn đã đọc',
        CONTINUES_RECENT_INTEREST: 'Tiếp nối chủ đề bạn quan tâm',
        TRENDING_IN_CATEGORY: 'Được quan tâm trong chuyên mục',
        POPULAR_NOW: 'Được nhiều người đọc lựa chọn',
    };
    const state = {
        personas: [], selected: null, history: [], articles: new Map(), recommendations: [],
        category: 'all', response: null, httpMs: null, controller: null, requestId: 0,
        updatedAt: null, modalArticle: null,
    };
    let toastTimer;
    const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[char]));
    const categoryName = (category) => categories[category] || category || 'Tổng hợp';
    const reasonLabel = (code) => reasons[code] || 'Một gợi ý dành cho bạn';
    const isGuest = (persona) => !persona?.id || persona.id === 'NEW_USER_COLD';
    const readerName = (persona) => isGuest(persona) ? 'Người đọc mới' : persona.name;
    const narrowScreen = window.matchMedia('(max-width: 760px)');
    $('reading-history').open = !narrowScreen.matches;
    narrowScreen.addEventListener('change', (event) => { $('reading-history').open = !event.matches; });

    function toast(message) {
        clearTimeout(toastTimer);
        $('toast').textContent = message;
        $('toast').dataset.visible = 'true';
        toastTimer = setTimeout(() => { $('toast').dataset.visible = 'false'; }, 3500);
    }

    function setConnection(status, label) {
        $('connection-status').dataset.state = status;
        $('connection-label').textContent = label;
    }

    async function checkReady() {
        try {
            const response = await fetch('/ready');
            setConnection(response.ok ? 'ready' : 'offline', response.ok ? 'Đã kết nối' : 'Chưa sẵn sàng');
        } catch {
            setConnection('offline', 'Mất kết nối');
        }
    }

    function initials(persona) {
        if (isGuest(persona)) return 'Mới';
        return readerName(persona).trim().split(/\s+/).slice(-2).map((word) => word[0]).join('').toUpperCase();
    }

    function renderReaders() {
        $('reader-list').innerHTML = state.personas.map((persona, index) => {
            const active = state.selected === persona;
            const interest = isGuest(persona) ? 'Bắt đầu với một bảng tin mới' : persona.category_pref || 'Hồ sơ tùy chọn';
            return `<button class="reader-option" type="button" data-reader="${index}" aria-pressed="${active}">
                <span class="reader-initials" aria-hidden="true">${escape(initials(persona))}</span>
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

    function selectReader(persona) {
        state.selected = persona;
        state.history = [...(persona.history_news_ids || [])].slice(-500);
        rememberArticles(persona.history);
        state.category = 'all';
        state.recommendations = [];
        state.response = null;
        $('feed-description').textContent = isGuest(persona)
            ? 'Bắt đầu với những bài được nhiều người đọc lựa chọn.'
            : `Một vài bài viết dành cho ${readerName(persona)}.`;
        renderReaders();
        renderHistory();
        renderDetails();
        fetchRecommendations();
    }

    function renderHistory() {
        $('history-count').textContent = state.history.length;
        $('history-context').textContent = state.selected
            ? `Lịch sử của ${readerName(state.selected)}` : 'Chọn người đọc để xem lịch sử.';
        $('clear-history-button').hidden = state.history.length === 0;
        if (!state.history.length) {
            $('history-list').innerHTML = '<li class="history-empty">Chưa có bài nào trong lịch sử. Thêm một bài từ bảng tin để thử những gợi ý khác.</li>';
            return;
        }
        // Show the eight most recent articles; send the entire history to the API.
        $('history-list').innerHTML = state.history.slice(-8).reverse().map((id, index) => {
            const title = state.articles.get(id)?.title || `Bài viết ${id}`;
            return `<li class="history-item"><span class="history-item-number">${index + 1}</span><p class="history-item-title" title="${escape(title)}">${escape(title)}</p><button class="remove-history" type="button" data-remove-history="${escape(id)}" aria-label="Bỏ ${escape(title)} khỏi lịch sử">×</button></li>`;
        }).join('');
    }

    function markStale() {
        $('feed-notice').hidden = false;
        $('feed-notice').dataset.kind = 'info';
        $('feed-notice').textContent = 'Bạn đã thay đổi lựa chọn. Bấm “Lấy gợi ý” để cập nhật bảng tin.';
    }

    function invalidateRequest() {
        state.requestId += 1;
        state.controller?.abort();
        $('recommend-button').disabled = !state.selected;
        $('recommend-button-label').textContent = 'Lấy gợi ý';
        $('article-list').setAttribute('aria-busy', 'false');
    }

    function updateHistory(nextHistory) {
        invalidateRequest();
        state.history = nextHistory;
        renderHistory();
        renderNews();
        markStale();
    }

    function addToHistory(item) {
        if (state.history.includes(item.news_id)) {
            toast('Bài này đã có trong lịch sử đọc.');
            return;
        }
        rememberArticles([{ ...item.article, news_id: item.news_id }]);
        updateHistory([...state.history, item.news_id].slice(-500));
        toast('Đã thêm bài vào lịch sử. Bấm “Lấy gợi ý” để cập nhật.');
    }

    function loadingState() {
        $('article-list').innerHTML = Array.from({ length: 5 }, () => '<div class="loading-article" aria-hidden="true"><div class="loading-line short"></div><div class="loading-line title"></div><div class="loading-line"></div><div class="loading-line" style="width:70%"></div></div>').join('');
        $('article-list').setAttribute('aria-busy', 'true');
    }

    function emptyState(title, description, retry = false) {
        $('article-list').innerHTML = `<div class="empty-state"><span class="empty-state-mark" aria-hidden="true">r.</span><h2>${escape(title)}</h2><p>${escape(description)}</p>${retry ? '<button class="secondary-button" type="button" id="retry-button">Thử lại</button>' : ''}</div>`;
        $('retry-button')?.addEventListener('click', fetchRecommendations);
    }

    function renderNotice() {
        const result = state.response;
        const notice = $('feed-notice');
        notice.hidden = true;
        if (!result) return;
        if (result.fallback_reason === 'COLD_USER') {
            notice.dataset.kind = 'info';
            notice.textContent = 'Hồ sơ này chưa được nhận diện. Gợi ý hiện tại dựa trên các nguồn có sẵn, bao gồm lịch sử bạn cung cấp.';
        } else if (result.degraded) {
            notice.dataset.kind = 'warning';
            notice.textContent = 'Một phần hệ thống phản hồi chậm hoặc chưa sẵn sàng. Bảng tin đang dùng kết quả dự phòng; bạn có thể thử lấy gợi ý lại.';
        } else return;
        notice.hidden = false;
    }

    async function fetchRecommendations() {
        if (!state.selected) return;
        state.controller?.abort();
        const controller = new AbortController();
        state.controller = controller;
        const requestId = ++state.requestId;
        $('recommend-button').disabled = true;
        $('recommend-button-label').textContent = 'Đang lấy tin…';
        $('feed-notice').hidden = true;
        $('category-nav').hidden = true;
        $('feed-caption').textContent = 'Đang tìm những bài phù hợp…';
        loadingState();
        const started = performance.now();
        try {
            const response = await fetch('/api/recommend-rich', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
                body: JSON.stringify({ user_id: isGuest(state.selected) ? null : state.selected.id,
                    history_news_ids: [...state.history], top_k: Number($('article-count').value),
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
            setConnection('ready', 'Đã kết nối');
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
                ? 'Hệ thống chưa nạp xong dữ liệu gợi ý. Bạn thử lại sau một chút nhé.'
                : 'Chưa kết nối được với hệ thống gợi ý. Kiểm tra server rồi thử lại nhé.';
            emptyState('Chưa lấy được bảng tin', description, true);
            renderDetails();
            checkReady();
        } finally {
            if (requestId === state.requestId) {
                $('recommend-button').disabled = false;
                $('recommend-button-label').textContent = 'Lấy gợi ý';
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
        const buttons = [['all', 'Tất cả', state.recommendations.length],
            ...Array.from(counts, ([key, count]) => [key, categoryName(key), count])];
        $('category-nav').innerHTML = buttons.map(([key, label, count]) => `<button class="category-button" type="button" data-category="${escape(key)}" aria-pressed="${state.category === key}">${escape(label)}<span>${count}</span></button>`).join('');
        $('category-nav').hidden = !state.recommendations.length;
    }

    function renderNews() {
        const items = state.recommendations.filter((item) => state.category === 'all' || (item.article?.category || 'general') === state.category);
        const time = state.updatedAt?.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
        $('feed-caption').textContent = state.response
            ? `${items.length} bài${state.category !== 'all' ? ` · ${categoryName(state.category)}` : ' dành cho bạn'}${time ? ` · Cập nhật ${time}` : ''}`
            : 'Chưa có lượt gợi ý';
        if (!items.length) {
            emptyState('Chưa có bài phù hợp', 'Thử đổi chuyên mục, thêm lịch sử đọc hoặc lấy gợi ý lại.');
            return;
        }
        $('article-list').innerHTML = items.map((item) => {
            const article = item.article || {};
            const saved = state.history.includes(item.news_id);
            return `<article class="news-article">
                <div class="article-meta"><span class="article-category">${escape(categoryName(article.category))}</span><span class="article-rank">${String(item.rank).padStart(2, '0')}</span></div>
                <h2 class="article-title"><button class="article-title-button" type="button" data-open-article="${escape(item.news_id)}">${escape(article.title || `Bài viết ${item.news_id}`)}</button></h2>
                <p class="article-abstract">${escape(article.abstract || 'Bài viết này chưa có nội dung tóm tắt.')}</p>
                <div class="article-bottom"><p class="article-reason">${escape(reasonLabel(item.reason_code))}</p>
                    <div class="article-actions"><button class="read-article-button" type="button" data-open-article="${escape(item.news_id)}">Đọc tóm tắt <span aria-hidden="true">→</span></button>
                        <button class="save-history-button" type="button" data-save-article="${escape(item.news_id)}" data-saved="${saved}">${saved ? '✓ Đã thêm' : '+ Thêm vào lịch sử'}</button></div>
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

    function openArticle(item) {
        state.modalArticle = item;
        const article = item.article || {};
        $('article-dialog-title').textContent = article.title || `Bài viết ${item.news_id}`;
        $('article-dialog-category').textContent = categoryName(article.category);
        $('article-dialog-reason').textContent = reasonLabel(item.reason_code);
        $('article-dialog-abstract').textContent = article.abstract || 'Bài viết này chưa có nội dung tóm tắt.';
        $('mark-read-button').textContent = state.history.includes(item.news_id) ? 'Đã có trong lịch sử' : 'Thêm vào lịch sử đọc';
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

    $('reader-list').addEventListener('click', (event) => {
        const button = event.target.closest('[data-reader]');
        if (button) selectReader(state.personas[Number(button.dataset.reader)]);
    });
    $('history-list').addEventListener('click', (event) => {
        const button = event.target.closest('[data-remove-history]');
        if (button) updateHistory(state.history.filter((id) => id !== button.dataset.removeHistory));
    });
    $('clear-history-button').addEventListener('click', () => {
        updateHistory([]);
        toast('Đã bỏ lịch sử của lượt trải nghiệm này.');
    });
    $('recommend-button').addEventListener('click', fetchRecommendations);
    for (const id of ['article-count', 'filter-seen']) {
        $(id).addEventListener('change', () => {
            if (state.selected) { invalidateRequest(); renderNews(); markStale(); }
        });
    }
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
        const persona = { id: id || null, name: 'Hồ sơ của bạn', category_pref: 'Lịch sử tự chọn', history_news_ids: history, history: [], custom: true };
        state.personas = state.personas.filter((entry) => !entry.custom);
        state.personas.push(persona);
        $('custom-dialog').close();
        selectReader(persona);
    });
    $('details-button').addEventListener('click', () => $('details-dialog').showModal());
    for (const dialog of document.querySelectorAll('dialog')) {
        dialog.querySelectorAll('[data-close-dialog]').forEach((button) => button.addEventListener('click', () => dialog.close()));
        dialog.addEventListener('click', (event) => {
            const bounds = dialog.getBoundingClientRect();
            if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
        });
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
            $('reader-list').innerHTML = '<p class="history-empty">Chưa tải được hồ sơ mẫu. Bạn vẫn có thể chọn “Dùng hồ sơ khác”.</p>';
            $('feed-caption').textContent = 'Chưa tải được hồ sơ';
            $('article-list').setAttribute('aria-busy', 'false');
            emptyState('Chọn cách bắt đầu', 'Dùng hồ sơ khác để nhập mã người đọc hoặc thử với lịch sử trống.');
        }
    }
    initialize();
});

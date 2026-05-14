    if request.method == 'POST':
        q = request.form.get('question', '').strip()
        if not q:
            return jsonify({'question': '', 'answer': 'Please enter a question.'}), 400

        try:
            ans = get_rag_answer(q)
        except Exception as e:
            ans = f"Error: {e}"

        user_id = session.get('user_id')
        if user_id:
            try:
                save_query(user_id, q, ans)
            except Exception:
                pass

        return jsonify({'question': q, 'answer': ans})

    return render_template('query.html')


# ── Crop Recommendation ──────────────────────────────────────────────────
@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        f = request.form
        try:
            X = pd.DataFrame([{
                "N":           float(f.get("nitrogen")),
                "P":           float(f.get("phosphorus")),
                "K":           float(f.get("potassium")),
                "temperature": float(f.get("temperature")),
                "humidity":    float(f.get("humidity")),
                "ph":          float(f.get("ph")),
                "rainfall":    float(f.get("rainfall")),
            }])
            crop           = model.predict(X)[0] if model else None
            normalized     = crop.strip().lower() if crop else None
            hindi_crop     = crop_translation.get(normalized, crop)
            conf           = round(max(model.predict_proba(X)[0]) * 100, 2) \
                             if (model and hasattr(model, "predict_proba")) else None
        except Exception as e:
            return render_template('recommendation.html', error=f"Error: {e}")

        return render_template(
            'recommendation.html',
            crop=crop, hindi_crop=hindi_crop, confidence=conf
        )
    return render_template('recommendation.html')


# ── Profile (view + update) ──────────────────────────────────────────────
@app.route('/profile', methods=['GET'])
@login_required
def profile():
    user_id = session.get('user_id')
    user    = get_user_by_id(user_id)
    return render_template('profile.html', user=user)


@app.route('/profile/update', methods=['POST'])
@login_required
def profile_update():
    user_id = session.get('user_id')
    fullname = request.form.get('fullname', '').strip()
    location = request.form.get('location', '').strip()

    if fullname or location:
        try:
            update_profile(user_id, fullname, location)
            if fullname:
                session['fullname'] = fullname
            flash("Profile updated successfully.", "success")
        except Exception as e:
            flash(f"Could not update profile: {e}", "error")
    else:
        flash("No changes submitted.", "info")

    return redirect(url_for('profile'))


# ── Admin ────────────────────────────────────────────────────────────────
@app.route('/admin')
@login_required
def admin():
    if session.get('user_id') != 1:
        flash("Admin access only.", "error")
        return redirect(url_for('home'))
    users = get_all_users()
    return render_template('admin.html', users=users)


# ── Error handlers ────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html'), 404

@app.errorhandler(500)
def server_error(e):
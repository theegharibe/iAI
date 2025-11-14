"""UI Components for Local AI Chat."""

import flet as ft


class UIComponents:
    """Reusable UI components."""
    
    @staticmethod
    def create_sidebar(app, theme: str):
        """Create sidebar with navigation."""
        suffix = ".light" if theme == "light" else ""
        
        logo_icon = app.assets.get_icon(f"logo{suffix}", 40)
        new_chat_icon = app.assets.get_icon(f"chat{suffix}", 32)
        question_icon = app.assets.get_icon(f"question{suffix}", 32)
        ai_train_icon = app.assets.get_icon(f"ai_train{suffix}", 32)
        rocket_icon = app.assets.get_icon(f"rocket{suffix}", 32)
        settings_icon = app.assets.get_icon(f"settings{suffix}", 32)
        info_icon = app.assets.get_icon(f"info{suffix}", 32)
        
        is_light = theme == "light"
        icon_color = ft.Colors.BLACK if is_light else ft.Colors.WHITE
        
        def create_icon_button(icon_data, icon_name, tooltip, on_click, fallback_icon=None, size=32):
            if icon_data:
                return ft.IconButton(
                    content=ft.Image(
                        src_base64=icon_data,
                        width=size,
                        height=size,
                        fit=ft.ImageFit.CONTAIN
                    ),
                    tooltip=tooltip,
                    on_click=on_click,
                    icon_color=icon_color
                )
            elif fallback_icon:
                return ft.IconButton(
                    icon=fallback_icon,
                    icon_size=size,
                    tooltip=tooltip,
                    on_click=on_click,
                    icon_color=icon_color
                )
            else:
                return None
        
        return ft.Container(
            content=ft.Column(
                controls=[
                    create_icon_button(logo_icon, "logo", "iAI Chat", None, ft.Icons.CHAT, 40),
                    
                    ft.Divider(height=20),
                    
                    create_icon_button(new_chat_icon, "chat", "New Chat", app.new_chat, ft.Icons.ADD_ROUNDED),
                    
                    create_icon_button(
                        question_icon,
                        "question",
                        "Ask Questions",
                        app.show_query_page,
                        ft.Icons.QUESTION_ANSWER
                    ),
                    
                    create_icon_button(
                        ai_train_icon,
                        "ai_train",
                        "Train Personal AI",
                        app.show_personal_ai_page,
                        ft.Icons.SCHOOL
                    ),
                    
                    create_icon_button(
                        rocket_icon,
                        "rocket",
                        "Use Your Model",
                        app.show_use_model_page,
                        ft.Icons.ROCKET_LAUNCH
                    ),
                    
                    ft.Container(expand=True),
                    
                    create_icon_button(settings_icon, "settings", "Settings", app.open_settings, ft.Icons.SETTINGS),
                    
                    create_icon_button(info_icon, "info", "About", app.show_about, ft.Icons.INFO_OUTLINE),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15
            ),
            width=80,
            padding=15,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.PRIMARY) if theme == 'dark' else ft.Colors.with_opacity(0.03, ft.Colors.PRIMARY)
        )
    
    @staticmethod
    def create_top_bar(app, theme: str):
        """Create top bar with model selector and chat management."""
        suffix = ".light" if theme == "light" else ""
        model_icon = app.assets.get_icon(f"model_one{suffix}", 32)
        chats_icon = app.assets.get_icon(f"chat{suffix}", 32)
        
        app.model_icon_container = ft.Container(
            content=ft.Image(src_base64=model_icon, width=32, height=32) if model_icon else ft.Icon(ft.Icons.MODEL_TRAINING),
            width=32,
            height=32,
        )
        
        app.toolbar_items = [
            app.model_icon_container,
            app.model_dropdown,
            ft.Container(expand=True),
            ft.IconButton(
                content=ft.Image(src_base64=chats_icon, width=32, height=32) if chats_icon else None,
                icon=ft.Icons.CHAT if not chats_icon else None,
                icon_size=32,
                tooltip="Manage Chats",
                on_click=app.toggle_chat_view
            ),
            app.status_text
        ]
        
        app.chat_items = [
            ft.IconButton(
                icon=ft.Icons.MENU,
                icon_size=32,
                tooltip="Show Main Menu",
                on_click=app.toggle_chat_view
            ),
            ft.Container(expand=True),
            ft.IconButton(
                icon=ft.Icons.ADD_COMMENT_ROUNDED,
                icon_size=32,
                tooltip="New Chat",
                on_click=app.new_chat
            ),
            ft.Text("", size=12, color=ft.Colors.GREY_500)
        ]
        
        return ft.Container(
            content=ft.Row(
                controls=[
                    app.model_icon_container,
                    app.model_dropdown,
                    ft.Container(expand=True),
                    app.status_text
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER
            ),
            padding=15,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.PRIMARY) if theme == 'dark' else ft.Colors.with_opacity(0.01, ft.Colors.PRIMARY)
        )
    
    @staticmethod
    def create_chat_toolbar(app, theme: str):
        """Create toolbar for chat management."""
        suffix = ".light" if theme == "light" else ""
        chat_buttons = []
        
        for chat in app.chat_histories:
            chat_id = chat["id"]
            chat_icon = app.assets.get_icon(f"chat{chat_id}{suffix}", 32)
            
            button = ft.IconButton(
                content=ft.Image(src_base64=chat_icon, width=32, height=32) if chat_icon else None,
                icon=ft.Icons.CHAT if not chat_icon else None,
                icon_size=32,
                tooltip=f"Chat {chat_id}",
                data=str(chat_id),
                on_click=lambda e: app.switch_to_chat(int(e.control.data)),
                style=ft.ButtonStyle(
                    color=ft.Colors.BLUE if chat_id == app.active_chat_id else None,
                    overlay_color=ft.Colors.with_opacity(0.1, ft.Colors.BLUE) if chat_id == app.active_chat_id else None
                )
            )
            chat_buttons.append(button)
        
        app.chat_toolbar = ft.Container(
            content=ft.Row(
                controls=chat_buttons,
                alignment=ft.MainAxisAlignment.START,
                spacing=10
            ),
            padding=ft.padding.symmetric(horizontal=15, vertical=5),
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.PRIMARY) if theme == 'dark' else ft.Colors.with_opacity(0.01, ft.Colors.PRIMARY)
        )
        
        return app.chat_toolbar
    
    @staticmethod
    def create_chat_area(app):
        """Create chat area with chat toolbar."""
        chat_buttons = []
        
        for chat in app.chat_histories:
            chat_id = chat["id"]
            suffix = ".light" if app.theme == "light" else ""
            chat_icon = app.assets.get_icon(f"chat{chat_id}{suffix}", 32)
            if not chat_icon:
                chat_icon = app.assets.get_icon(f"chat{suffix}", 32)
            
            button = ft.IconButton(
                content=ft.Image(src_base64=chat_icon, width=32, height=32) if chat_icon else None,
                icon=ft.Icons.CHAT if not chat_icon else None,
                icon_size=32,
                tooltip=f"Chat {chat_id}",
                data=str(chat_id),
                on_click=lambda e: app.switch_to_chat(int(e.control.data)),
                style=ft.ButtonStyle(
                    color=ft.Colors.BLUE if chat_id == app.active_chat_id else None,
                    overlay_color=ft.Colors.with_opacity(0.1, ft.Colors.BLUE) if chat_id == app.active_chat_id else None
                )
            )
            chat_buttons.append(button)
        
        app.chat_toolbar = ft.Row(
            controls=chat_buttons,
            alignment=ft.MainAxisAlignment.START,
            spacing=10,
            visible=app.is_chat_view
        )
        
        chat_area = ft.Column(
            controls=[
                ft.Container(
                    content=app.chat_toolbar,
                    padding=ft.padding.symmetric(horizontal=15, vertical=5),
                    bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.PRIMARY) if app.theme == 'dark' 
                            else ft.Colors.with_opacity(0.01, ft.Colors.PRIMARY)
                ),
                ft.Container(
                    content=app.chat_view,
                    expand=True,
                    bgcolor=ft.Colors.TRANSPARENT
                )
            ],
            spacing=0,
            expand=True
        )
        
        return chat_area
    
    @staticmethod
    def create_input_area(app, theme: str):
        """Create input area."""
        suffix = ".light" if theme == "light" else ""
        
        send_icon = app.assets.get_icon(f"send{suffix}", 28)
        clear_icon = app.assets.get_icon(f"clear{suffix}", 28)
        
        return ft.Container(
            content=ft.Row(
                controls=[
                    app.input_field,
                    ft.IconButton(
                        content=ft.Image(src_base64=clear_icon, width=28, height=28) if clear_icon else None,
                        icon=ft.Icons.DELETE_OUTLINE if not clear_icon else None,
                        icon_size=28,
                        tooltip="Clear Chat",
                        on_click=app.clear_chat
                    ),
                    ft.IconButton(
                        content=ft.Image(src_base64=send_icon, width=28, height=28) if send_icon else None,
                        icon=ft.Icons.SEND_ROUNDED if not send_icon else None,
                        icon_size=28,
                        tooltip="Send Message",
                        on_click=app.send_message
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER
            ),
            padding=ft.padding.symmetric(horizontal=20, vertical=15),
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.PRIMARY) if theme == 'dark' else ft.Colors.with_opacity(0.01, ft.Colors.PRIMARY)
        )
    
    @staticmethod
    def create_settings_popup(app):
        """Create settings popup overlay with monochrome styling."""
        def close_popup(_):
            if popup in app.page.overlay:
                app.page.overlay.remove(popup)
            if overlay in app.page.overlay:
                app.page.overlay.remove(overlay)
            app.page.update()

        def save_settings(_):
            app.save_settings()
            close_popup(_)

        def handle_context_change(e):
            value = int(e.control.value)
            setattr(app, 'context_length', value)
            helper_text.color = ft.Colors.GREY_500 if value <= 5000 else ft.Colors.WHITE
            e.control.active_color = ft.Colors.WHITE if value <= 5000 else ft.Colors.GREY_400
            helper_text.update()
            e.control.update()

        overlay = ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.4, ft.Colors.BLACK),
            on_click=close_popup
        )

        helper_text = ft.Text(
            "Memory usage (recommended: 1024-5000)", 
            size=12, 
            color=ft.Colors.GREY_500 if app.context_length <= 5000 else ft.Colors.WHITE
        )

        settings_content = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Text("Settings", size=24, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        ft.Container(
                            content=ft.IconButton(
                                icon=ft.Icons.CLOSE,
                                icon_color=ft.Colors.GREY_400,
                                icon_size=20,
                                on_click=close_popup
                            ),
                            margin=ft.margin.only(right=15, left=30)
                        )
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                
                ft.Divider(color=ft.Colors.GREY_800),

                ft.Text("Model Configuration", size=18, weight=ft.FontWeight.BOLD),
                
                ft.Container(
                    content=ft.Column([
                        ft.Text("Temperature", size=16),
                        ft.Container(
                            content=ft.Slider(
                                min=0.0,
                                max=2.0,
                                value=app.temperature,
                                divisions=20,
                                label="{value}",
                                active_color=ft.Colors.WHITE,
                                inactive_color=ft.Colors.GREY_800,
                                on_change=lambda e: setattr(app, 'temperature', e.control.value)
                            ),
                            margin=ft.margin.only(right=50)
                        ),
                        ft.Text(
                            "[Lower = focused responses] [Higher = creative responses]", 
                            size=12, 
                            color=ft.Colors.GREY_400
                        ),
                    ]),
                    margin=ft.margin.only(left=10)
                ),
                
                ft.Container(height=20),
                
                ft.Container(
                    content=ft.Column([
                        ft.Text("Context Length", size=16),
                        ft.Container(
                            content=ft.Slider(
                                min=1024,
                                max=30000,
                                value=min(app.context_length, 30000),
                                divisions=29,
                                label="{value}",
                                on_change=handle_context_change,
                                active_color=ft.Colors.WHITE if app.context_length <= 5000 
                                    else ft.Colors.GREY_400,
                                inactive_color=ft.Colors.GREY_800,
                            ),
                            margin=ft.margin.only(right=50)
                        ),
                        helper_text
                    ]),
                    margin=ft.margin.only(left=10)
                ),
                
                ft.Divider(color=ft.Colors.GREY_800),

                ft.Text("Appearance", size=18, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Switch(
                                value=app.theme == 'dark',
                                on_change=app.toggle_theme,
                                active_color=ft.Colors.WHITE,
                                inactive_thumb_color=ft.Colors.GREY_400,
                            ),
                            ft.Text("Dark Theme", size=16),
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    margin=ft.margin.symmetric(horizontal=10, vertical=10)
                ),

                ft.Container(
                    content=ft.IconButton(
                        content=ft.Container(
                            content=ft.Image(
                                src_base64=app.assets.get_icon(f"saveas{'.light' if app.theme == 'light' else ''}", 64),
                                width=90,
                                height=48,
                                fit=ft.ImageFit.CONTAIN,
                                repeat=ft.ImageRepeat.NO_REPEAT,
                                gapless_playback=True,
                            ) if app.assets.get_icon(f"saveas{'.light' if app.theme == 'light' else ''}", 64) else None,
                        ),
                        icon=ft.Icons.SAVE_ALT if not app.assets.get_icon(f"saveas{'.light' if app.theme == 'light' else ''}", 64) else None,
                        tooltip="Save Changes",
                        on_click=save_settings
                    ),
                    alignment=ft.alignment.center,
                    margin=ft.margin.only(top=20)
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=20
        )

        popup = ft.Container(
            content=ft.Card(
                content=ft.Container(
                    content=ft.Container(
                        content=settings_content,
                        padding=ft.padding.only(right=35, left=10),
                        expand=True
                    ),
                    padding=ft.padding.only(left=20, right=20, top=20, bottom=30),
                )
            ),
            width=550,
            height=700,
            bgcolor=ft.Colors.with_opacity(0.95, ft.Colors.ON_SURFACE_VARIANT),
            border_radius=10,
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
            animate_opacity=300,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            alignment=ft.alignment.center
        )

        return overlay, popup
    
    @staticmethod
    def create_about_popup(app):
        """Create about popup overlay."""
        def close_popup(_):
            if popup in app.page.overlay:
                app.page.overlay.remove(popup)
            if overlay in app.page.overlay:
                app.page.overlay.remove(overlay)
            app.page.update()

        overlay = ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.4, ft.Colors.BLACK),
            on_click=close_popup
        )

        about_content = """ iAi 1.0.0


iAI is an innovative project designed to empower users by allowing them to upload personal files directly to an advanced artificial intelligence system.
\n\nThe AI performs a meticulous analysis of the file contents, seamlessly integrating the extracted insights into its core knowledge base.\n\n
Once the analysis is complete, users can interact with the AI in versatile ways—such as asking targeted questions about the data, generating custom insights, or executing specialized tasks beyond basic queries.\n\n
This persistent integration ensures that the uploaded information becomes a permanent extension of the AI's understanding, fostering more contextual, personalized, and efficient interactions over time.

© 2024 - Open Source
"""
        content_stack = ft.Stack([
            ft.Container(
                bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
                width=800,
                height=800,
                margin=ft.margin.all(-40),
                alignment=ft.alignment.center,
            ),
            ft.Column(
                controls=[
                    ft.Text("About", size=24, weight=ft.FontWeight.BOLD),
                    ft.Text(about_content, size=13),
                    ft.Container(height=20),
                    ft.ElevatedButton("Close", on_click=close_popup)
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=30
            ),
            ft.Container(
                content=ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_color=ft.Colors.GREY_400,
                    icon_size=20,
                    on_click=close_popup,
                ),
                alignment=ft.alignment.top_right,
                margin=ft.margin.only(top=-10, right=-10),
            ),
        ])

        popup = ft.Container(
            content=ft.Card(
                content=ft.Container(
                    content=content_stack,
                    padding=40
                )
            ),
            width=400,
            height=600,
            bgcolor=ft.Colors.with_opacity(0.95, ft.Colors.ON_SURFACE_VARIANT),
            border_radius=10,
            animate=ft.Animation(300, ft.AnimationCurve.BOUNCE_OUT),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        )

        return overlay, popup
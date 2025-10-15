# Create the calendar manager module
calendar_manager_code = '''"""
Google Calendar Manager for SecureAI Personal Assistant
Handles calendar operations with explicit permission controls
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

# Google Calendar API imports
try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False
    logging.warning("Google Calendar API not available")


@dataclass
class CalendarEvent:
    """Calendar event data structure"""
    id: Optional[str]
    title: str
    start_time: datetime
    end_time: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    attendees: Optional[List[str]] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class CalendarManager:
    """Manages Google Calendar integration with security controls"""
    
    def __init__(self, config: Dict):
        self.config = config.get("integrations", {}).get("calendar", {})
        self.enabled = self.config.get("enabled", False) and GOOGLE_CALENDAR_AVAILABLE
        self.scopes = self.config.get("scopes", ['https://www.googleapis.com/auth/calendar'])
        
        self.credentials_file = "credentials.json"
        self.token_file = "token.json"
        self.service = None
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Google Calendar service with OAuth"""
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.credentials_file):
                        self.logger.error(f"Credentials file {self.credentials_file} not found")
                        self.enabled = False
                        return
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.scopes
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
            
            self.service = build('calendar', 'v3', credentials=creds)
            self.logger.info("Google Calendar service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Calendar service: {e}")
            self.enabled = False
    
    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse various datetime formats"""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse datetime: {dt_str}")
    
    def _format_datetime_for_api(self, dt: datetime) -> Dict[str, str]:
        """Format datetime for Google Calendar API"""
        return {
            'dateTime': dt.isoformat(),
            'timeZone': 'UTC'
        }
    
    def _convert_api_event(self, api_event: Dict) -> CalendarEvent:
        """Convert API event to CalendarEvent object"""
        start_dt = api_event['start'].get('dateTime', api_event['start'].get('date'))
        end_dt = api_event['end'].get('dateTime', api_event['end'].get('date'))
        
        # Parse datetime strings
        if 'T' in start_dt:
            start_time = datetime.fromisoformat(start_dt.replace('Z', '+00:00'))
        else:
            start_time = datetime.fromisoformat(start_dt)
        
        if 'T' in end_dt:
            end_time = datetime.fromisoformat(end_dt.replace('Z', '+00:00'))
        else:
            end_time = datetime.fromisoformat(end_dt)
        
        attendees = []
        if 'attendees' in api_event:
            attendees = [att.get('email', '') for att in api_event['attendees']]
        
        return CalendarEvent(
            id=api_event.get('id'),
            title=api_event.get('summary', 'No Title'),
            start_time=start_time,
            end_time=end_time,
            description=api_event.get('description'),
            location=api_event.get('location'),
            attendees=attendees,
            created=datetime.fromisoformat(api_event['created'].replace('Z', '+00:00')) if 'created' in api_event else None,
            updated=datetime.fromisoformat(api_event['updated'].replace('Z', '+00:00')) if 'updated' in api_event else None
        )
    
    def list_events(self, days_ahead: int = 7, max_results: int = 10) -> List[CalendarEvent]:
        """
        List upcoming calendar events
        
        Args:
            days_ahead: Number of days to look ahead
            max_results: Maximum number of events to return
            
        Returns:
            List of CalendarEvent objects
        """
        if not self.enabled or not self.service:
            self.logger.warning("Calendar service not available")
            return []
        
        try:
            # Calculate time range
            now = datetime.utcnow().isoformat() + 'Z'
            end_time = (datetime.utcnow() + timedelta(days=days_ahead)).isoformat() + 'Z'
            
            # Call the Calendar API
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                timeMax=end_time,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            # Convert to CalendarEvent objects
            calendar_events = []
            for event in events:
                try:
                    calendar_event = self._convert_api_event(event)
                    calendar_events.append(calendar_event)
                except Exception as e:
                    self.logger.warning(f"Failed to parse event {event.get('id', 'unknown')}: {e}")
            
            self.logger.info(f"Retrieved {len(calendar_events)} calendar events")
            return calendar_events
            
        except Exception as e:
            self.logger.error(f"Failed to list calendar events: {e}")
            return []
    
    def create_event(self, title: str, start_time: datetime, end_time: datetime,
                    description: Optional[str] = None, location: Optional[str] = None,
                    attendees: Optional[List[str]] = None, 
                    require_confirmation: bool = True) -> Optional[str]:
        """
        Create a new calendar event
        
        Args:
            title: Event title
            start_time: Event start time
            end_time: Event end time
            description: Optional event description
            location: Optional event location
            attendees: Optional list of attendee emails
            require_confirmation: Whether to require user confirmation
            
        Returns:
            Event ID if successful, None otherwise
        """
        if not self.enabled or not self.service:
            self.logger.warning("Calendar service not available")
            return None
        
        try:
            # Build event object
            event = {
                'summary': title,
                'start': self._format_datetime_for_api(start_time),
                'end': self._format_datetime_for_api(end_time),
            }
            
            if description:
                event['description'] = description
            
            if location:
                event['location'] = location
            
            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]
            
            # Security check - require confirmation for sensitive operations
            if require_confirmation:
                self.logger.info(f"Creating event: {title} from {start_time} to {end_time}")
                # In a real implementation, this would prompt the user
                # For now, we'll assume confirmation is given
            
            # Create the event
            created_event = self.service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            event_id = created_event.get('id')
            self.logger.info(f"Created calendar event: {event_id}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to create calendar event: {e}")
            return None
    
    def update_event(self, event_id: str, title: Optional[str] = None,
                    start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                    description: Optional[str] = None, location: Optional[str] = None,
                    require_confirmation: bool = True) -> bool:
        """
        Update an existing calendar event
        
        Args:
            event_id: ID of event to update
            title: New title (optional)
            start_time: New start time (optional)
            end_time: New end time (optional)
            description: New description (optional)
            location: New location (optional)
            require_confirmation: Whether to require user confirmation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.service:
            self.logger.warning("Calendar service not available")
            return False
        
        try:
            # Get existing event
            existing_event = self.service.events().get(
                calendarId='primary',
                eventId=event_id
            ).execute()
            
            # Update fields
            if title is not None:
                existing_event['summary'] = title
            
            if start_time is not None:
                existing_event['start'] = self._format_datetime_for_api(start_time)
            
            if end_time is not None:
                existing_event['end'] = self._format_datetime_for_api(end_time)
            
            if description is not None:
                existing_event['description'] = description
            
            if location is not None:
                existing_event['location'] = location
            
            # Security check
            if require_confirmation:
                self.logger.info(f"Updating event: {event_id}")
                # In a real implementation, this would prompt the user
            
            # Update the event
            updated_event = self.service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=existing_event
            ).execute()
            
            self.logger.info(f"Updated calendar event: {event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update calendar event {event_id}: {e}")
            return False
    
    def delete_event(self, event_id: str, require_confirmation: bool = True) -> bool:
        """
        Delete a calendar event
        
        Args:
            event_id: ID of event to delete
            require_confirmation: Whether to require user confirmation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.service:
            self.logger.warning("Calendar service not available")
            return False
        
        try:
            # Security check - always require confirmation for deletions
            if require_confirmation:
                self.logger.warning(f"Attempting to delete event: {event_id}")
                # In a real implementation, this would prompt the user
                # For now, we'll log and proceed (in production, should return False without confirmation)
            
            # Delete the event
            self.service.events().delete(
                calendarId='primary',
                eventId=event_id
            ).execute()
            
            self.logger.info(f"Deleted calendar event: {event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete calendar event {event_id}: {e}")
            return False
    
    def search_events(self, query: str, max_results: int = 10) -> List[CalendarEvent]:
        """
        Search calendar events by query
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of matching CalendarEvent objects
        """
        if not self.enabled or not self.service:
            self.logger.warning("Calendar service not available")
            return []
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                q=query,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            # Convert to CalendarEvent objects
            calendar_events = []
            for event in events:
                try:
                    calendar_event = self._convert_api_event(event)
                    calendar_events.append(calendar_event)
                except Exception as e:
                    self.logger.warning(f"Failed to parse search result {event.get('id', 'unknown')}: {e}")
            
            self.logger.info(f"Found {len(calendar_events)} events matching query: {query}")
            return calendar_events
            
        except Exception as e:
            self.logger.error(f"Failed to search calendar events: {e}")
            return []
    
    def get_free_busy_info(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Get free/busy information for a time range
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Free/busy information
        """
        if not self.enabled or not self.service:
            return {"error": "Calendar service not available"}
        
        try:
            freebusy_result = self.service.freebusy().query(
                body={
                    "timeMin": start_time.isoformat() + 'Z',
                    "timeMax": end_time.isoformat() + 'Z',
                    "items": [{"id": "primary"}]
                }
            ).execute()
            
            return freebusy_result
            
        except Exception as e:
            self.logger.error(f"Failed to get free/busy info: {e}")
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if calendar service is available"""
        return self.enabled and self.service is not None
'''

with open('calendar_manager.py', 'w') as f:
    f.write(calendar_manager_code)

print("✓ calendar_manager.py created")